"""devops/proxy/check_domain.py: does a domain reveal its owner, and does it answer? (Made-up replies.)"""

import importlib.util
import ssl
import sys
import urllib.error
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
spec = importlib.util.spec_from_file_location("check_domain", ROOT / "devops" / "proxy" / "check_domain.py")
cd = importlib.util.module_from_spec(spec)
sys.modules["check_domain"] = cd
spec.loader.exec_module(cd)


def contact(role, **fields):
    items = [["version", {}, "text", "4.0"]]
    for name, value in fields.items():
        if name == "adr":
            items.append(["adr", {}, "text", ["", "", value[0], value[1], "", "", "US"]])
        else:
            items.append([name, {}, "text", value])
    return {"roles": [role], "vcardArray": ["vcard", items]}


HIDDEN = {
    "events": [
        {"eventAction": "registration", "eventDate": "2026-11-02T10:00:00Z"},
        {"eventAction": "expiration", "eventDate": "2027-11-02T10:00:00Z"},
    ],
    "status": ["client transfer prohibited", "active"],
    "nameservers": [{"ldhName": "NS-1.AWSDNS-01.ORG"}],
    "entities": [
        {"roles": ["registrar"], "vcardArray": ["vcard", [["fn", {}, "text", "Amazon Registrar, Inc."]]]},
        contact(
            "registrant",
            fn="REDACTED FOR PRIVACY",
            org="Data Protected",
            email="Please query the RDAP service",
        ),
        contact(
            "administrative", fn="Redacted for Privacy", adr=("REDACTED FOR PRIVACY", "REDACTED FOR PRIVACY")
        ),
        contact("technical", fn="Domains By Proxy, LLC"),
    ],
}


def levels(findings):
    return [f.level for f in findings]


class TestContacts:
    def test_hidden_and_redacted_contacts_are_fine(self):
        findings = cd.contact_findings(HIDDEN)
        assert levels(findings) == ["OK", "OK", "OK"] and "registrant contact is hidden" in str(findings[0])

    def test_a_shown_name_email_phone_or_street_is_flagged_and_masked(self):
        exposed = {
            "entities": [
                contact(
                    "registrant",
                    fn="Jane Doe",
                    org="Doe Lab",
                    email="jane.doe@example.org",
                    tel="+1.5551234567",
                ),
                contact("administrative", adr=("12 Elm Street", "Springfield")),
            ]
        }
        findings = cd.contact_findings(exposed)
        assert levels(findings) == ["WARN", "WARN"]
        text = " ".join(str(f) for f in findings)
        assert "fn Ja***" in text and "email j***@example.org" in text and "tel +1***" in text
        assert (
            "Jane Doe" not in text and "jane.doe" not in text and "5551234567" not in text
        )  # masked in the report
        assert "street 12***" in text and "Springfield" not in text

    def test_country_alone_does_not_reveal_anyone(self):
        assert levels(cd.contact_findings({"entities": [contact("registrant", adr=("", ""))]})) == ["OK"]

    def test_no_contacts_at_all_is_fine_and_other_roles_are_ignored(self):
        registrar_only = {
            "entities": [
                {"roles": ["registrar"], "vcardArray": ["vcard", [["fn", {}, "text", "Some Registrar"]]]}
            ]
        }
        assert levels(cd.contact_findings(registrar_only)) == ["OK"]
        assert "no registrant" in str(cd.contact_findings({})[0])

    def test_mask(self):
        assert (
            cd.mask("a.b@example.org") == "a***@example.org"
            and cd.mask("ab") == "***"
            and cd.mask("abcdef") == "ab***"
        )


class TestRegistration:
    def test_registrar_dates_lock_and_name_servers(self):
        text = " ".join(str(f) for f in cd.registration_findings(HIDDEN))
        assert "Amazon Registrar, Inc." in text and "registered 2026-11-02, expires 2027-11-02" in text
        assert "transfer lock is on" in text and "ns-1.awsdns-01.org" in text

    def test_a_missing_lock_and_missing_details_are_noticed_and_the_statuses_are_shown(self):
        findings = cd.registration_findings({"status": ["active"]})
        assert "registrar: not shown" in str(findings[0]) and "?" in str(findings[1])
        assert any(f.level == "INFO" and "status shown by the registry: active" in f.text for f in findings)
        warning = next(f for f in findings if f.level == "WARN")
        assert (
            "no transfer lock" in warning.text and "Registered domains" in warning.text
        )  # says where to fix it
        assert "status shown by the registry: none" in str(cd.registration_findings({})[2])
        assert not any("name servers at the registry" in f.text for f in findings)

    def test_a_brand_new_domain_without_a_lock_is_information_not_a_warning(self):
        new = cd.registration_findings({"status": ["add period"]})
        assert not any(f.level == "WARN" for f in new)
        text = next(f.text for f in new if "no transfer lock" in f.text)
        assert "brand new" in text and "60 day" not in text and "Transfer lock" in text
        assert any("status shown by the registry: add period" in f.text for f in new)
        locked = cd.registration_findings({"status": ["add period", "client transfer prohibited"]})
        assert any(
            f.level == "OK" and "transfer lock is on" in f.text for f in locked
        )  # the lock wins when shown

    def test_check_registration_reads_rdap(self):
        findings = cd.check_registration("confoundsearch.org", lambda url: HIDDEN)
        assert "WARN" not in levels(findings) and len(findings) > 4

    @pytest.mark.parametrize(
        "error,wanted",
        [
            (urllib.error.HTTPError("u", 404, "nf", {}, None), "not registered"),
            (urllib.error.HTTPError("u", 500, "err", {}, None), "HTTP 500"),
            (urllib.error.URLError("no route"), "could not read the registration"),
            (ValueError("bad json"), "could not read the registration"),
        ],
        ids=["not-found", "server-error", "no-route", "bad-json"],  # pytest cannot name an HTTPError on 3.9
    )
    def test_rdap_failures_are_warnings_not_crashes(self, error, wanted):
        def fetch(url):
            raise error

        (finding,) = cd.check_registration("x.org", fetch)
        assert finding.level == "WARN" and wanted in finding.text


class TestDns:
    @staticmethod
    def answers(ns=(), a=(), aaaa=()):
        def fetch(url):
            kind = "NS" if "type=NS" in url else "AAAA" if "type=AAAA" in url else "A"
            data, code = {"NS": (ns, 2), "A": (a, 1), "AAAA": (aaaa, 28)}[kind]
            return {"Answer": [{"type": code, "data": f"{v}."} for v in data]}

        return fetch

    def test_route53_name_servers_and_a_host_that_resolves(self):
        fetch = self.answers(ns=["ns-1.awsdns-01.org", "ns-2.awsdns-02.com"], a=["1.2.3.4"])
        findings = cd.check_dns("confoundsearch.org", "confoundsearch.org", fetch)
        assert [f.level for f in findings] == ["OK", "OK"]
        assert (
            "Route 53 (2 name servers)" in findings[0].text
            and "confoundsearch.org resolves (1.2.3.4)" in findings[1].text
        )

    def test_not_resolving_before_deploy_is_expected_not_a_warning(self):
        findings = cd.check_dns("x.org", "api.x.org", self.answers(ns=["ns-1.awsdns-01.org"]))
        assert findings[1].level == "INFO" and "expected until the service is deployed" in findings[1].text

    def test_name_servers_elsewhere_or_none_are_warned(self):
        assert cd.route53_findings(["ns1.otherhost.com", "ns-1.awsdns-01.org"])[0].level == "WARN"
        assert "not (only) in Route 53" in cd.route53_findings(["ns1.otherhost.com"])[0].text
        assert "not be delegated" in cd.route53_findings([])[0].text

    def test_dns_over_https_failure_is_a_warning(self):
        def fetch(url):
            raise urllib.error.URLError("offline")

        findings = cd.check_dns("x.org", "api.x.org", fetch)
        assert findings[-1].level == "WARN" and "could not ask DNS over HTTPS" in findings[-1].text


CERT = {"subject": ((("commonName", "confoundsearch.org"),),), "notAfter": "Nov  2 00:00:00 2027 GMT"}


class TestHttps:
    def test_a_working_service(self):
        findings = cd.check_https("confoundsearch.org", lambda host: CERT, lambda url: {"ok": True})
        assert (
            levels(findings) == ["OK", "INFO", "OK"]
            and "valid certificate for confoundsearch.org" in findings[0].text
        )

    def test_a_service_that_answers_but_is_not_healthy_or_does_not_answer_health(self):
        assert cd.check_https("h", lambda host: CERT, lambda url: {"ok": False})[-1].level == "WARN"

        def broken(url):
            raise urllib.error.URLError("refused")

        assert "did not answer" in cd.check_https("h", lambda host: CERT, broken)[-1].text

    @pytest.mark.parametrize("error", [OSError("no route"), ssl.SSLError("bad cert")])
    def test_not_answering_yet_is_information_not_a_warning(self, error):
        def refuse(host):
            raise error

        (finding,) = cd.check_https("api.x.org", refuse)
        assert finding.level == "INFO" and "expected before deploy" in finding.text


class TestFetch:
    def test_fetch_json_reads_a_local_server_and_doh_parses_answers(self):
        import http.server
        import threading

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.end_headers()
                self.wfile.write(
                    b'{"Answer": [{"type": 2, "data": "NS-1.AWSDNS-01.ORG."}, {"type": 16, "data": "txt"}]}'
                )

            def log_message(self, *args):
                pass

        server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            data = cd.fetch_json(f"http://127.0.0.1:{server.server_port}/")
        finally:
            server.shutdown()
            server.server_close()
        assert cd.doh(lambda url: data, "x.org", "NS") == [
            "ns-1.awsdns-01.org"
        ]  # the TXT answer is not a name


def test_main_prints_a_report_and_exits_by_whether_anything_needs_a_look(monkeypatch, capsys):
    monkeypatch.setattr(cd, "check_registration", lambda domain: [cd.Finding("OK", "fine")])
    monkeypatch.setattr(cd, "check_dns", lambda domain, host: [cd.Finding("INFO", f"host {host}")])
    monkeypatch.setattr(cd, "check_https", lambda host: [cd.Finding("OK", "answers")])
    assert cd.main(["confoundsearch.org"]) == 0
    out = capsys.readouterr().out
    assert "host confoundsearch.org" in out and "nothing to look at" in out
    monkeypatch.setattr(cd, "check_https", lambda host: [cd.Finding("WARN", "bad")])
    assert cd.main(["confoundsearch.org", "--host", "svc.confoundsearch.org"]) == 1
    assert "1 thing(s) to look at" in capsys.readouterr().out
