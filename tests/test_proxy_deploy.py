"""devops/proxy: the Lambda zip and the template that deploys it. The template is read, never run, here."""

import importlib.util
import re
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
PROXY = ROOT / "devops" / "proxy"
spec = importlib.util.spec_from_file_location("proxy_build", PROXY / "build.py")
build = importlib.util.module_from_spec(spec)
sys.modules["proxy_build"] = build
spec.loader.exec_module(build)


def test_the_zip_holds_the_handler_and_the_package_and_no_bytecode(tmp_path):
    out = build.build(tmp_path / "lambda.zip")
    names = zipfile.ZipFile(out).namelist()
    assert "handler.py" in names and "confoundsearch/data/vectors.py" in names
    assert "confoundsearch/data/keyword_service.py" in names
    assert not [n for n in names if "__pycache__" in n or n.endswith(".pyc")]
    assert out.stat().st_size < 5_000_000  # a Lambda zip has a 50 MB limit; ours is tiny


def test_the_same_files_build_the_same_zip(tmp_path):
    first = build.build(tmp_path / "a.zip").read_bytes()
    second = build.build(tmp_path / "b" / "c.zip").read_bytes()
    assert first == second  # so a re-deploy of unchanged code changes nothing


def test_the_file_list_is_sorted_and_skips_what_it_should(tmp_path):
    package = tmp_path / "pkg"
    (package / "sub" / "__pycache__").mkdir(parents=True)
    (package / "b.py").parent.mkdir(exist_ok=True)
    for name in ("b.py", "a.py", "sub/c.py", "sub/__pycache__/c.cpython-313.pyc", "d.pyc"):
        (package / name).write_text("x")
    handler = tmp_path / "handler.py"
    handler.write_text("x")
    listed = [name for _, name in build.files(package, handler)]
    assert listed == ["handler.py", "pkg/a.py", "pkg/b.py", "pkg/sub/c.py"]


def test_main_writes_the_zip_where_asked(tmp_path, capsys):
    assert build.main([str(tmp_path / "x" / "l.zip")]) == 0
    assert (tmp_path / "x" / "l.zip").exists() and "wrote" in capsys.readouterr().out


# -- the template -------------------------------------------------------------------------------------------


def load_template():
    yaml = pytest.importorskip("yaml")

    class Loader(yaml.SafeLoader):
        pass

    # CloudFormation's short forms (!Ref, !Sub, ...) are not plain YAML: read them as {tag: value}
    def construct(loader, suffix, node):
        if isinstance(node, yaml.ScalarNode):
            value = loader.construct_scalar(node)
        elif isinstance(node, yaml.SequenceNode):
            value = loader.construct_sequence(node, deep=True)
        else:
            value = loader.construct_mapping(node, deep=True)
        return {suffix: value}

    Loader.add_multi_constructor("!", construct)
    return yaml.load((PROXY / "template.yaml").read_text(encoding="utf-8"), Loader=Loader)


def test_the_template_exposes_exactly_the_six_routes():
    events = load_template()["Resources"]["Function"]["Properties"]["Events"]
    routes = {(e["Properties"]["Method"], e["Properties"]["Path"]) for e in events.values()}
    assert routes == {
        ("GET", "/v1/health"),
        ("GET", "/v1/manifest"),
        ("POST", "/v1/neighbors"),
        ("POST", "/v1/papers"),
        ("POST", "/v1/fetch"),
        ("POST", "/v1/query"),
    }  # no catch-all, no ANY


def test_the_template_limits_the_rate_and_the_concurrency_and_alerts_on_spend():
    resources = load_template()["Resources"]
    api = resources["Api"]["Properties"]["DefaultRouteSettings"]
    assert "ThrottlingRateLimit" in api and "ThrottlingBurstLimit" in api
    assert "ReservedConcurrentExecutions" in resources["Function"]["Properties"]
    assert resources["Budget"]["Type"] == "AWS::Budgets::Budget" and "Condition" in resources["Budget"]


def test_the_function_may_read_its_two_keys_and_do_nothing_else():
    statements = load_template()["Resources"]["Function"]["Properties"]["Policies"][0]["Statement"]
    assert [s["Action"] for s in statements] == ["ssm:GetParameter"]
    resources = str(statements[0]["Resource"])
    assert "PineconeKeyParameter" in resources and "ElasticKeyParameter" in resources
    assert "*" not in resources  # named parameters only


def test_the_template_holds_no_secret_and_no_openai_key():
    text = (PROXY / "template.yaml").read_text(encoding="utf-8")
    assert not re.search(r"sk-[A-Za-z0-9]{10,}|pcsk_|pc-[A-Za-z0-9]{20,}", text)
    environment = load_template()["Resources"]["Function"]["Properties"]["Environment"]["Variables"]
    assert not [name for name in environment if "OPENAI" in name]  # callers bring their own key
    assert (
        "ELASTIC_KEY_PARAMETER" in environment and "ELASTIC_API_KEY" not in environment
    )  # a name, never a key
    assert "PINECONE_API_KEY" not in environment
    parameters = load_template()["Parameters"]
    assert "AlertEmail" in parameters and parameters["AlertEmail"]["Default"] == ""  # no address is baked in
    assert parameters["ElasticUrl"]["Default"] == ""  # no private endpoint is baked in either


def test_the_stack_reports_the_address_to_put_in_the_package():
    assert "ApiUrl" in load_template()["Outputs"]


# -- the permanent address ----------------------------------------------------------------------------------


def test_the_custom_domain_is_optional_and_off_by_default():
    template = load_template()
    parameters = template["Parameters"]
    assert parameters["DomainName"]["Default"] == "" and parameters["HostedZoneId"]["Default"] == ""
    assert parameters["CertificateArn"]["Default"] == ""
    for name in ("Certificate", "CustomDomain", "CustomDomainMapping", "DnsRecord"):
        assert "Condition" in template["Resources"][name], name  # nothing is made unless a domain is given
    for output in ("PermanentUrl", "DnsTarget"):
        assert "Condition" in template["Outputs"][output]


def test_the_domain_is_regional_with_modern_tls_and_points_at_the_api():
    resources = load_template()["Resources"]
    config = resources["CustomDomain"]["Properties"]["DomainNameConfigurations"][0]
    assert config["EndpointType"] == "REGIONAL" and config["SecurityPolicy"] == "TLS_1_2"
    mapping = resources["CustomDomainMapping"]["Properties"]
    assert mapping["Stage"] == "$default" and mapping["ApiId"] == {"Ref": "Api"}
    assert (
        resources["CustomDomainMapping"]["DependsOn"] == "ApiApiGatewayDefaultStage"
    )  # the stage must exist first


def test_the_certificate_is_made_and_validated_by_dns_only_when_there_is_a_zone_and_no_certificate_given():
    template = load_template()
    conditions = template["Conditions"]
    assert conditions["MakeCertificate"] == {
        "And": [
            {"Condition": "HasDomain"},
            {"Condition": "HasZone"},
            {"Not": [{"Condition": "HasCertificateArn"}]},
        ]
    }
    certificate = template["Resources"]["Certificate"]["Properties"]
    assert certificate["ValidationMethod"] == "DNS"
    chosen = template["Resources"]["CustomDomain"]["Properties"]["DomainNameConfigurations"][0][
        "CertificateArn"
    ]
    assert chosen["If"][0] == "HasCertificateArn"  # a certificate you supply wins over making one


def test_the_dns_record_is_an_alias_to_the_domain_and_made_only_with_a_zone():
    template = load_template()
    assert template["Conditions"]["MakeDnsRecord"] == {
        "And": [{"Condition": "HasDomain"}, {"Condition": "HasZone"}]
    }
    record = template["Resources"]["DnsRecord"]["Properties"]
    assert record["Type"] == "A" and "AliasTarget" in record


def test_closing_the_generated_address_can_never_happen_without_a_domain():
    template = load_template()
    assert template["Conditions"]["CloseDefaultAddress"] == {
        "And": [{"Condition": "HasDomain"}, {"Equals": [{"Ref": "DisableDefaultAddress"}, "true"]}]
    }
    api = template["Resources"]["Api"]["Properties"]
    assert api["DisableExecuteApiEndpoint"]["If"][0] == "CloseDefaultAddress"
    assert (
        template["Parameters"]["DisableDefaultAddress"]["Default"] == "false"
    )  # open until you choose to close it


def test_the_code_comes_from_a_bucket_the_deployer_names_so_the_console_can_deploy_it():
    template = load_template()
    parameters = template["Parameters"]
    assert (
        "Default" not in parameters["CodeBucket"] and "Default" not in parameters["CodeKey"]
    )  # asked every time
    code = template["Resources"]["Function"]["Properties"]["CodeUri"]
    assert code == {"Bucket": {"Ref": "CodeBucket"}, "Key": {"Ref": "CodeKey"}}
    assert "build/lambda.zip" not in (PROXY / "template.yaml").read_text(
        encoding="utf-8"
    )  # no local path is left


@pytest.mark.parametrize(
    "name,accepted,refused",
    [
        (
            "CodeBucket",
            ["confoundsearch-deploy-8f3k2", "my.bucket-1"],
            ["", "ab", "UPPER", "-x-", "has space", "a" * 64],
        ),
        ("CodeKey", ["lambda.zip", "builds/lambda-2.zip"], ["", "zip", ".zip", "lambda.txt"]),
        ("HostedZoneId", ["", "Z0074055A32U67PYRPJO"], ["z123", "Z 1", "/hostedzone/Z123", "abc"]),
    ],
)
def test_the_form_refuses_a_blank_or_malformed_code_location_and_zone_id(name, accepted, refused):
    """CloudFormation applies these to the whole value, so the form flags a wrong entry next to the field."""
    parameter = load_template()["Parameters"][name]

    def allowed(value):
        within = parameter.get("MinLength", 0) <= len(value) <= parameter.get("MaxLength", 10**9)
        return within and re.fullmatch(parameter["AllowedPattern"], value) is not None

    assert all(allowed(v) for v in accepted) and not any(allowed(v) for v in refused)
    assert parameter["ConstraintDescription"]  # and the message says what is wanted
