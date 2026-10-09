"""Suricata-style signature catalog and alert-emulation layer.

The MachineLearningCSV release of CIC-IDS2017 distributes CICFlowMeter flow
records without timestamps or endpoint addresses. This module constructs the
alert stream for the pilot study:

1. labeled flows are mapped to signature families with per-family firing
   probabilities and per-flow detection probability (signature fidelity);
2. flow timestamps are drawn deterministically (seeded) inside the documented
   attack windows of schedule.ATTACKS;
3. endpoint addresses are drawn from the documented topology;
4. benign flows fire false-positive alerts at a calibrated per-flow rate with
   their own signature families, giving the noise parameter of the study.

The released harness accepts real Suricata EVE JSON as a drop-in replacement
(ingest.load_eve), which is the path for full-scale PCAP replay; the pilot
uses this emulation so the study stays within public-data bounds.
"""

import hashlib
import random
from dataclasses import dataclass, field

from . import schedule

# ---------------------------------------------------------------------------
# Signature catalog: (sid_prefix, family, message template, severity)
# Each family has >=2 signature variants so that the same incident can produce
# heterogeneous signature identifiers (the regime where static signature-keyed
# aggregation fragments).
# ---------------------------------------------------------------------------

SIGNATURES = {
    "FTP-Patator": [
        ("ET SCAN Multiple FTP Login Attempts (patator)", "BRUTE-FTP", 2),
        ("ET SCAN FTP Brute Force Activity Detected", "BRUTE-FTP", 2),
    ],
    "SSH-Patator": [
        ("ET SCAN Multiple SSH Login Attempts (patator)", "BRUTE-SSH", 2),
        ("ET SCAN Possible SSH Brute Force", "BRUTE-SSH", 2),
    ],
    "DoS slowloris": [
        ("ET DOS HTTP Slowloris Partial Header Flood", "DOS-SLOWLORIS", 3),
        ("ET DOS Slow HTTP Attack (partial requests)", "DOS-SLOWLORIS", 3),
    ],
    "DoS Slowhttptest": [
        ("ET DOS Slow HTTP POST Attack (slowhttptest)", "DOS-SLOWHTTP", 3),
        ("ET DOS Prolonged HTTP Header Occupation", "DOS-SLOWHTTP", 3),
    ],
    "DoS Hulk": [
        ("ET DOS HTTP POST Flood (hulk pattern)", "DOS-HULK", 3),
        ("ET DOS Web Server Massive GET Flood", "DOS-HULK", 3),
    ],
    "DoS GoldenEye": [
        ("ET DOS GoldenEye HTTP DoS Flood", "DOS-GOLDENEYE", 3),
        ("ET DOS HTTP Keep-Alive Flood Pattern", "DOS-GOLDENEYE", 3),
    ],
    "Heartbleed": [
        ("ET EXPLOIT OpenSSL Heartbleed Malformed TLS Record", "EXPLOIT", 4),
        ("ET EXPLOIT Possible Heartbeat Response Abuse", "EXPLOIT", 4),
    ],
    "Web Attack - Brute Force": [
        ("ET WEB_SERVER Web Application Brute Force (damnsmall)", "WEB-BRUTE", 3),
        ("ET WEB_SERVER Repeated Auth Failures on Login Form", "WEB-BRUTE", 2),
    ],
    "Web Attack - XSS": [
        ("ET WEB_SERVER Possible XSS in Request Parameter", "WEB-XSS", 3),
        ("ET WEB_SERVER Script Tag Injection Attempt", "WEB-XSS", 3),
    ],
    "Web Attack - SQL Injection": [
        ("ET WEB_SERVER Possible SQL Injection Attempt (union select)", "WEB-SQLI", 3),
        ("ET WEB_SERVER SQL Error String in Response Path", "WEB-SQLI", 3),
    ],
    "Infiltration": [
        ("ET TROJAN Metasploit Meterpreter Activity", "TROJAN", 4),
        ("ET TROJAN Cool Disk MAC Malware Beacon", "TROJAN", 4),
        ("ET POLICY Outbound Connection to Dropbox API (payload fetch)", "POLICY-EXFIL", 2),
    ],
    "Bot": [
        ("ET TROJAN Botnet ARES C2 Check-in", "BOTNET", 4),
        ("ET TROJAN ARES P2P Command Traffic", "BOTNET", 4),
    ],
    "PortScan": [
        ("ET SCAN Nmap SYN Sweep From Single Host", "SCAN", 2),
        ("ET SCAN Closed Port Response Flood (sT/sF variant)", "SCAN", 2),
        ("ET SCAN Version Probe Pattern (sV)", "SCAN", 2),
    ],
    "DDoS": [
        ("ET DOS LOIC HTTP Flood From Multiple Hosts", "DDOS", 4),
        ("ET DOS Mass GET Request Storm", "DDOS", 4),
    ],
}

# False-positive signature families for benign traffic
FP_SIGNATURES = [
    ("ET POLICY Inbound Connection to Unusual High Port", "POLICY", 1),
    ("ET INFO Outbound DNS Query to Newly Observed Domain", "INFO", 1),
    ("ET SCAN Possible Nmap OS Fingerprint Probe", "SCAN", 1),
    ("ET POLICY HTTP Client Contains Suspicious User-Agent", "POLICY", 1),
    ("ET INFO P2P Activity on Non-Standard Port", "INFO", 1),
    ("ET POLICY Outbound TLS to Unsigned External Host", "POLICY", 1),
]

# Per-flow firing probability for labeled flows (signature fidelity)
P_DETECT = {
    "FTP-Patator": 0.90, "SSH-Patator": 0.90,
    "DoS slowloris": 0.85, "DoS Slowhttptest": 0.85,
    "DoS Hulk": 0.90, "DoS GoldenEye": 0.90,
    "Heartbleed": 0.85,
    "Web Attack - Brute Force": 0.70, "Web Attack - XSS": 0.60,
    "Web Attack - SQL Injection": 0.55,
    "Infiltration": 0.70, "Bot": 0.75,
    "PortScan": 0.22, "DDoS": 0.85,
}


@dataclass
class Alert:
    alert_id: int
    ts: float                    # unix epoch seconds
    src: str
    dst: str
    dst_port: int
    proto: str
    sid: str                     # stable per-signature id, e.g. ET-100012
    family: str                  # signature family label
    text: str                    # full alert message (what an analyst reads)
    severity: int
    is_attack: bool
    gt_incident: str = ""        # ground-truth incident id ("" for benign)

    def to_dict(self):
        return dict(alert_id=self.alert_id, ts=self.ts, src=self.src,
                    dst=self.dst, dst_port=self.dst_port, proto=self.proto,
                    sid=self.sid, family=self.family, text=self.text,
                    severity=self.severity, is_attack=self.is_attack,
                    gt_incident=self.gt_incident)


def _sid_for(template: str) -> str:
    h = int(hashlib.md5(template.encode()).hexdigest()[:6], 16) % 900000 + 100000
    return f"ET-{h}"


def _alert_text(template: str, src: str, dst: str, port: int, proto: str,
                extra: str) -> str:
    base = f"{template} [{proto} {src} -> {dst}:{port}]"
    return f"{base} {extra}".strip()


def attack_alert_for(label: str, ts: float, rng: random.Random, port: int,
                     proto: str, src: str, dst: str, incident: str,
                     alert_id: int, flow_hint: str) -> Alert:
    """Build one alert for a labeled attack flow."""
    variants = SIGNATURES[label]
    template, family, sev = variants[rng.randrange(len(variants))]
    return Alert(
        alert_id=alert_id, ts=ts, src=src, dst=dst, dst_port=port,
        proto=proto, sid=_sid_for(template), family=family, severity=sev,
        text=_alert_text(template, src, dst, port, proto, flow_hint),
        is_attack=True, gt_incident=incident,
    )


def fp_alert_for(ts: float, rng: random.Random, alert_id: int) -> Alert:
    """Build one false-positive alert on benign traffic."""
    hosts = schedule.TOPOLOGY
    src = rng.choice(hosts["clients"] + [hosts["firewall_in"]])
    dst = rng.choice(hosts["clients"] + [hosts["web_server"], hosts["ubuntu12"]])
    port = rng.choice([80, 443, 21, 22, 445, 3389, 8080, 53, 123])
    proto = rng.choice(["TCP", "TCP", "TCP", "UDP"])
    template, family, sev = FP_SIGNATURES[rng.randrange(len(FP_SIGNATURES))]
    return Alert(
        alert_id=alert_id, ts=ts, src=src, dst=dst, dst_port=port,
        proto=proto, sid=_sid_for(template), family=family, severity=sev,
        text=_alert_text(template, src, dst, port, proto, ""),
        is_attack=False, gt_incident="",
    )
