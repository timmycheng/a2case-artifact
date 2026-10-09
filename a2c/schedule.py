"""Documented CIC-IDS2017 attack schedule and topology.

Sources: Sharafaldin et al., ICISSP 2018 (dataset paper) and the official
dataset page at https://www.unb.ca/cic/datasets/ids-2017.html (retrieved
September 2026). These windows are the ground-truth basis for the
session-to-incident mapping.
"""

from datetime import datetime

# Network topology (documented addresses)
TOPOLOGY = {
    "firewall_out": "205.174.165.80",
    "firewall_in": "172.16.0.1",
    "attacker_kali": "205.174.165.73",
    "attacker_win_69": "205.174.165.69",
    "attacker_win_70": "205.174.165.70",
    "attacker_win_71": "205.174.165.71",
    "web_server": "192.168.10.50",
    "ubuntu12": "192.168.10.51",
    "win_vista": "192.168.10.8",
    "win_mac": "192.168.10.25",
    "dns_dc": "192.168.10.3",
    "clients": ["192.168.10.5", "192.168.10.8", "192.168.10.9",
                "192.168.10.12", "192.168.10.14", "192.168.10.15",
                "192.168.10.16", "192.168.10.17", "192.168.10.19",
                "192.168.10.25"],
}

DAY_FILES = {
    "monday": "Monday-WorkingHours.pcap_ISCX.csv",
    "tuesday": "Tuesday-WorkingHours.pcap_ISCX.csv",
    "wednesday": "Wednesday-workingHours.pcap_ISCX.csv",
    "thursday_am": "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "thursday_pm": "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "friday_am": "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "friday_ps": "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "friday_ddos": "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
}

# Attack windows: label-normalized name -> (day_key, start, end,
# attacker ip(s), victim ip(s), dst_port)
def _t(day, hm_end):
    """Build datetime on the 2017 attack-week calendar."""
    dates = {
        "tue": "2017-07-04", "wed": "2017-07-05", "thu": "2017-07-06",
        "fri": "2017-07-07",
    }
    day, hm = hm_end.split(" ", 1) if False else (None, None)
    raise NotImplementedError


def _win(day_date, start_hm, end_hm):
    s = datetime.strptime(f"{day_date} {start_hm}", "%Y-%m-%d %H:%M")
    e = datetime.strptime(f"{day_date} {end_hm}", "%Y-%m-%d %H:%M")
    return s, e


ATTACKS = {
    # Tuesday, July 4 — brute force
    "FTP-Patator": dict(day="tuesday", window=_win("2017-07-04", "09:20", "10:20"),
                        src=["205.174.165.73"], dst="192.168.10.50", port=21),
    "SSH-Patator": dict(day="tuesday", window=_win("2017-07-04", "14:00", "15:00"),
                        src=["205.174.165.73"], dst="192.168.10.50", port=22),
    # Wednesday, July 5 — DoS + Heartbleed
    "DoS slowloris": dict(day="wednesday", window=_win("2017-07-05", "09:47", "10:10"),
                          src=["205.174.165.73"], dst="192.168.10.50", port=80),
    "DoS Slowhttptest": dict(day="wednesday", window=_win("2017-07-05", "10:14", "10:35"),
                             src=["205.174.165.73"], dst="192.168.10.50", port=80),
    "DoS Hulk": dict(day="wednesday", window=_win("2017-07-05", "10:43", "11:00"),
                     src=["205.174.165.73"], dst="192.168.10.50", port=80),
    "DoS GoldenEye": dict(day="wednesday", window=_win("2017-07-05", "11:10", "11:23"),
                          src=["205.174.165.73"], dst="192.168.10.50", port=80),
    "Heartbleed": dict(day="wednesday", window=_win("2017-07-05", "15:12", "15:32"),
                       src=["205.174.165.73"], dst="192.168.10.51", port=444),
    # Thursday, July 6 AM — web attacks
    "Web Attack - Brute Force": dict(day="thursday_am", window=_win("2017-07-06", "09:20", "10:00"),
                                     src=["205.174.165.73"], dst="192.168.10.50", port=80),
    "Web Attack - XSS": dict(day="thursday_am", window=_win("2017-07-06", "10:15", "10:35"),
                             src=["205.174.165.73"], dst="192.168.10.50", port=80),
    "Web Attack - SQL Injection": dict(day="thursday_am", window=_win("2017-07-06", "10:40", "10:42"),
                                       src=["205.174.165.73"], dst="192.168.10.50", port=80),
    # Thursday, July 6 PM — infiltration phase 1 (multi-stage)
    "Infiltration": dict(day="thursday_pm", window=_win("2017-07-06", "14:19", "15:45"),
                         src=["205.174.165.73"], dst="192.168.10.8", port=445),
    # Friday, July 7
    "Bot": dict(day="friday_am", window=_win("2017-07-07", "10:02", "11:02"),
                src=["205.174.165.73"], dst="192.168.10.15", port=443),
    "PortScan": dict(day="friday_ps", window=_win("2017-07-07", "13:55", "15:29"),
                     src=["205.174.165.73"], dst="192.168.10.50", port=0),
    "DDoS": dict(day="friday_ddos", window=_win("2017-07-07", "15:56", "16:16"),
                 src=["205.174.165.69", "205.174.165.70", "205.174.165.71"],
                 dst="192.168.10.50", port=80),
}

# Label spellings found in the CSV files (incl. the en-dash variants)
LABEL_ALIASES = {
    "BENIGN": "BENIGN",
    "FTP-Patator": "FTP-Patator",
    "SSH-Patator": "SSH-Patator",
    "DoS slowloris": "DoS slowloris",
    "DoS Slowhttptest": "DoS Slowhttptest",
    "DoS Hulk": "DoS Hulk",
    "DoS GoldenEye": "DoS GoldenEye",
    "Heartbleed": "Heartbleed",
    "Web Attack \u2013 Brute Force": "Web Attack - Brute Force",
    "Web Attack - Brute Force": "Web Attack - Brute Force",
    "Web Attack \u2013 XSS": "Web Attack - XSS",
    "Web Attack - XSS": "Web Attack - XSS",
    "Web Attack \u2013 Sql Injection": "Web Attack - SQL Injection",
    "Web Attack - SQL Injection": "Web Attack - SQL Injection",
    "Infiltration": "Infiltration",
    "Bot": "Bot",
    "PortScan": "PortScan",
    "DDoS": "DDoS",
}

# Infiltration is one label spanning three documented phases; the Friday
# morning file carries the tail of the infiltration series plus Botnet ARES.
INFILTRATION_PHASES = [
    ("infil-meta", _win("2017-07-06", "14:19", "14:35")),
    ("infil-cooldisk", _win("2017-07-06", "14:53", "15:00")),
    ("infil-dropbox", _win("2017-07-06", "15:04", "15:45")),
]
