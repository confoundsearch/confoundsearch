"""Check a domain before it goes into the package: does it identify you, and does it answer?

    python devops/proxy/check_domain.py confoundsearch.org
    python devops/proxy/check_domain.py confoundsearch.org --host svc.confoundsearch.org   # if a subdomain

It asks three public sources and prints what each says:

  1. RDAP (the modern WHOIS) for the registration: the registrar, the dates, and above all whether the
     registrant, administrative and technical contacts show a name, an organisation, an email, a phone number
     or a street address, or are hidden. Anything shown is printed (partly masked) so that you can judge it.
  2. DNS, over HTTPS (Cloudflare), for the name servers: are they Route 53's (awsdns)? and does the host
     resolve?
  3. HTTPS to the host: does it answer, and what certificate does it present?

Standard library only; it needs the internet. OK is good, WARN needs a look, INFO is just so that you know.
Before the service is deployed, "does not resolve yet" for the service host is expected.
"""

from __future__ import annotations

import argparse
import json
import re
import socket
import ssl
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable

RDAP = "https://rdap.org/domain/"
DOH = "https://cloudflare-dns.com/dns-query"
CONTACT_ROLES = ("registrant", "administrative", "technical", "billing", "abuse")
HIDDEN = re.compile(
    r"redact|privacy|private|withheld|not disclosed|proxy|whoisguard|data protected|gdpr|"
    r"contact (the )?registrar|see .*rdap|query the (rdap|whois)|select contact|anonym|identity protect",
    re.I,
)
Fetch = Callable[[str], Any]  # url -> parsed JSON


@dataclass(frozen=True)
class Finding:
    level: str  # OK, WARN or INFO
    text: str

    def __str__(self) -> str:
        return f"{self.level:<5} {self.text}"


def fetch_json(url: str, accept: str = "application/json", timeout: float = 20.0) -> Any:
    request = urllib.request.Request(url, headers={"Accept": accept, "User-Agent": "confoundsearch-check/1"})
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        return json.loads(response.read())


def mask(value: str) -> str:
    """Enough to recognise a value, not all of it: 'jane.doe@example.org' -> 'j***@example.org'."""
    if "@" in value:
        local, _, host = value.partition("@")
        return f"{local[:1]}***@{host}"
    return value[:2] + "***" if len(value) > 4 else "***"


def vcard_values(entity: dict[str, Any]) -> dict[str, list[str]]:
    """{'fn': [...], 'org': [...], 'email': [...], 'tel': [...], 'street': [...]} from an RDAP contact."""
    found: dict[str, list[str]] = {"fn": [], "org": [], "email": [], "tel": [], "street": []}
    for item in (entity.get("vcardArray") or [None, []])[1]:
        name, value = item[0], item[3] if len(item) > 3 else ""
        if name in ("fn", "org", "email", "tel") and isinstance(value, str) and value.strip():
            found[name].append(value.strip())
        if name == "adr" and isinstance(value, list):
            found["street"] += [str(part) for part in value[2:4] if str(part).strip()]  # street and city only
    return found


def contact_findings(data: dict[str, Any]) -> list[Finding]:
    """What the contacts reveal, role by role. A hidden or redacted value is fine; anything else is shown."""
    out: list[Finding] = []
    seen_role = False
    for entity in data.get("entities", []):
        roles = [r for r in entity.get("roles", []) if r in CONTACT_ROLES]
        if not roles:
            continue
        seen_role = True
        shown = [
            f"{kind} {mask(v)}"
            for kind, values in vcard_values(entity).items()
            for v in values
            if not HIDDEN.search(v)
        ]
        label = "/".join(roles)
        if shown:
            out.append(Finding("WARN", f"{label} contact shows: {', '.join(shown)}  (is this you?)"))
        else:
            out.append(Finding("OK", f"{label} contact is hidden or redacted"))
    if not seen_role:
        out.append(Finding("OK", "the registration lists no registrant, admin or technical contacts at all"))
    return out


def registration_findings(data: dict[str, Any]) -> list[Finding]:
    out = []
    registrar = next(
        (vcard_values(e)["fn"] for e in data.get("entities", []) if "registrar" in e.get("roles", [])), []
    )
    out.append(Finding("INFO", f"registrar: {registrar[0] if registrar else 'not shown'}"))
    dates = {e.get("eventAction"): str(e.get("eventDate", ""))[:10] for e in data.get("events", [])}
    out.append(
        Finding(
            "INFO", f"registered {dates.get('registration', '?')}, expires {dates.get('expiration', '?')}"
        )
    )
    status = [str(x) for x in data.get("status", [])]
    out.append(Finding("INFO", "status shown by the registry: " + (", ".join(status) or "none")))
    squashed = [x.replace(" ", "").lower() for x in status]
    if any("transferprohibited" in x for x in squashed):
        out.append(Finding("OK", "transfer lock is on"))
    elif any("addperiod" in x for x in squashed):
        # under five days old: the registry has not added its statuses yet, and a domain this new cannot be
        # transferred in any case (there is a 60 day rule)
        out.append(
            Finding(
                "INFO",
                "no transfer lock shown yet, but this domain is brand new (registry status: add period), so "
                "it cannot be transferred anyway. Enable the lock in the Route 53 console (Registered "
                "domains, your domain, Transfer lock) and check again in a few days",
            )
        )
    else:
        out.append(
            Finding(
                "WARN",
                "the registry shows no transfer lock. For a Route 53 domain look under Registered domains, "
                "your domain, Transfer lock, and enable it",
            )
        )
    names = [n.get("ldhName", "").lower() for n in data.get("nameservers", [])]
    if names:
        out.append(Finding("INFO", "name servers at the registry: " + ", ".join(sorted(names))))
    return out


def route53_findings(name_servers: list[str]) -> list[Finding]:
    if not name_servers:
        return [Finding("WARN", "no name servers found: the domain may not be delegated yet")]
    if all("awsdns" in n for n in name_servers):
        return [Finding("OK", f"DNS is in Route 53 ({len(name_servers)} name servers)")]
    return [Finding("WARN", "DNS is not (only) in Route 53: " + ", ".join(sorted(name_servers)))]


def doh(fetch: Fetch, name: str, kind: str) -> list[str]:
    reply = fetch(f"{DOH}?{urllib.parse.urlencode({'name': name, 'type': kind})}")
    return sorted(
        a["data"].rstrip(".").lower() for a in reply.get("Answer", []) if a.get("type") in (1, 2, 5, 28)
    )


def check_registration(
    domain: str, fetch: Fetch = lambda u: fetch_json(u, "application/rdap+json")
) -> list[Finding]:
    try:
        data = fetch(RDAP + domain)
    except urllib.error.HTTPError as err:
        text = (
            "is not registered (or the registry has no record)"
            if err.code == 404
            else f"gave HTTP {err.code}"
        )
        return [Finding("WARN", f"RDAP {text}")]
    except (urllib.error.URLError, OSError, ValueError) as err:
        return [Finding("WARN", f"could not read the registration (RDAP): {err}")]
    return [*registration_findings(data), *contact_findings(data)]


def check_dns(
    domain: str, host: str, fetch: Fetch = lambda u: fetch_json(u, "application/dns-json")
) -> list[Finding]:
    out: list[Finding] = []
    try:
        out += route53_findings(doh(fetch, domain, "NS"))
        addresses = doh(fetch, host, "A") + doh(fetch, host, "AAAA")
    except (urllib.error.URLError, OSError, ValueError) as err:
        return [*out, Finding("WARN", f"could not ask DNS over HTTPS: {err}")]
    if addresses:
        out.append(Finding("OK", f"{host} resolves ({', '.join(addresses[:3])})"))
    else:
        out.append(Finding("INFO", f"{host} does not resolve yet (expected until the service is deployed)"))
    return out


def certificate(host: str, timeout: float = 15.0) -> dict[str, Any]:
    context = ssl.create_default_context()
    with socket.create_connection((host, 443), timeout) as raw, context.wrap_socket(
        raw, server_hostname=host
    ) as tls:
        return tls.getpeercert() or {}


def check_https(
    host: str, get_cert: Callable[[str], dict[str, Any]] = certificate, fetch: Fetch | None = None
) -> list[Finding]:
    try:
        cert = get_cert(host)
    except (OSError, ssl.SSLError) as err:
        return [
            Finding(
                "INFO", f"https://{host} does not answer yet: {type(err).__name__} (expected before deploy)"
            )
        ]
    subject = dict(x[0] for x in cert.get("subject", ()))
    out = [
        Finding("OK", f"https://{host} answers with a valid certificate for {subject.get('commonName', '?')}")
    ]
    out.append(Finding("INFO", f"certificate expires {cert.get('notAfter', '?')}"))
    try:
        health = (fetch or fetch_json)(f"https://{host}/v1/health")
        out.append(Finding("OK" if health.get("ok") else "WARN", f"/v1/health says {health}"))
    except (urllib.error.URLError, OSError, ValueError) as err:
        out.append(Finding("WARN", f"/v1/health did not answer: {err}"))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("domain", help="the domain you own, e.g. confoundsearch.org")
    parser.add_argument("--host", help="the name the service answers on (default: the domain itself)")
    args = parser.parse_args(argv)
    host = args.host or args.domain
    findings = [
        Finding("INFO", f"checking {args.domain} (service host {host})"),
        *check_registration(args.domain),
        *check_dns(args.domain, host),
        *check_https(host),
    ]
    print("\n".join(str(f) for f in findings))
    warned = [f for f in findings if f.level == "WARN"]
    print(f"\n{len(warned)} thing(s) to look at" if warned else "\nnothing to look at")
    return 1 if warned else 0


if __name__ == "__main__":
    sys.exit(main())
