import pytest
from app import redaction
from app.redaction import redact, restore


def test_personal_email_is_replaced_and_role_address_is_kept():
    r = redact("Write to jane.doe@example.com or to privacy@kestrel.example")
    assert "jane.doe@example.com" not in r.text
    assert "[EMAIL_1]" in r.text
    assert "privacy@kestrel.example" in r.text          # a function, not a person
    assert r.counts == {"EMAIL": 1}


def test_same_value_gets_the_same_placeholder():
    r = redact("a@b.org wrote to c@d.org, then a@b.org again")
    assert r.text.count("[EMAIL_1]") == 2 and r.text.count("[EMAIL_2]") == 1
    assert r.counts == {"EMAIL": 2}


@pytest.mark.parametrize("phone", ["+33 6 12 34 56 78", "06 12 34 56 78", "+1 (415) 555-0132", "415-555-0132 0"])
def test_phone_numbers_are_replaced(phone):
    assert "[PHONE_1]" in redact(f"Call me on {phone} tonight").text


@pytest.mark.parametrize("not_phone", [
    "error KST-2044 happened", "on 2026-11-14 at 18:32", "server 192.168.100.200 is down",
    "version 12.4.1", "invoice 20261114123456", "from 01/02/2026",
])
def test_codes_dates_ips_and_versions_are_not_phones(not_phone):
    assert redact(not_phone).text == not_phone


def test_ip_addresses_are_not_personal_data_here():
    assert redact("Allow 10.0.0.12 and 203.0.113.7").text == "Allow 10.0.0.12 and 203.0.113.7"


def test_names_need_a_cue():
    assert "Jane Doe" not in redact("Hello, my name is Jane Doe and I need help").text
    assert "Smith" not in redact("Please ask Dr. Smith").text
    assert redact("Contact Data Integrations").text == "Contact Data Integrations"   # a team, not a person
    assert redact("I am Kestrel Analytics").text == "I am Kestrel Analytics"


def test_a_name_inside_a_quote():
    r = redact('He said: "my name is Tomasz Wójcik, call me"')
    assert "Tomasz" not in r.text and "[PERSON_1]" in r.text


def test_known_names_file(tmp_path, monkeypatch):
    names = tmp_path / "names.txt"
    names.write_text("# the people of my documents\nIrina Petrova\n", encoding="utf-8")
    monkeypatch.setattr(redaction, "REDACTION_NAMES_FILE", names)
    assert redact("Ask Irina Petrova about it").text == "Ask [PERSON_1] about it"


def test_restore_puts_the_originals_back_json_safe():
    r = redact("mail jane@example.com")
    assert restore('{"answer": "write to [EMAIL_1]"}', r.mapping, json_safe=True) == '{"answer": "write to jane@example.com"}'
    r2 = redact('my name is Jane "JJ" Doe')
    assert restore('"[PERSON_1]"', r2.mapping, json_safe=True).count('"') >= 2


def test_disabled_changes_nothing():
    assert redact("jane@example.com", enabled=False).text == "jane@example.com"
