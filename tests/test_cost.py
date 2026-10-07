from whichapiapi.cost.engine import call_cost, monthly_cost
from whichapiapi.schema import Conditions, LoadProfile, Offer, Price, Provenance
from whichapiapi.schema.offer import Billing, OffPeak


def test_call_cost_with_cache():
    p = Price(input_per_1m=2.0, output_per_1m=10.0, cache_read_per_1m=0.2)
    assert call_cost(p, 1_000_000, 100_000, cached_tokens=500_000) == 1.0 + 0.1 + 1.0
    assert call_cost(Price(), 10, 10) is None


def test_monthly_cost_conditions():
    o = Offer(
        model="m",
        channel="c",
        price=Price(input_per_1m=1.0, output_per_1m=4.0),
        conditions=Conditions(batch_discount=0.5, off_peak=OffPeak(window="16:30-00:30", discount=0.5)),
        billing=Billing(topup_fee=0.05),
        provenance=Provenance(source="test"),
    )
    prof = LoadProfile(requests_per_month=1000, input_tokens=1000, output_tokens=250, batch_share=0.5)
    b = monthly_cost(o, prof)
    per_req_list = 1000 / 1e6 * 1.0 + 250 / 1e6 * 4.0  # 0.002
    assert b.list_per_month == round(per_req_list * 1000, 4)
    assert b.per_request == round(per_req_list * 0.75 * 1.05, 8)
    assert any("batch" in n for n in b.notes)
