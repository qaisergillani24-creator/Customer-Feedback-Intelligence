import itertools
import json

import pandas as pd
from google.genai import types

import agent as A
import tools as T
from analytics import add_categories

failed = []


def check(name, ok):
    print(("PASS  " if ok else "FAIL  ") + name)
    if not ok:
        failed.append(name)


def raises(fn, exc, *a, **k):
    try:
        fn(*a, **k)
    except exc:
        return True
    except Exception:
        return False
    return False


df = pd.DataFrame({
    "customer_id": [f"CID{i:03d}" for i in range(9)],
    "product": ["P1", "P1", "P1", "P1", "P2", "P2", "P2", "P2", "P2"],
    "feedback": [
        "Great battery and fast charging",
        "Love the display, very sharp",
        "The delivery was late and the box was damaged",
        "Late delivery again, very disappointing",
        "The screen is broken after one week",
        "Poor quality and a broken screen",
        "Package arrived late",
        "It does the job, nothing special",
        "Unclear text here",
    ],
    "sentiment": ["positive", "positive", "negative", "negative", "negative",
                  "negative", "negative", "neutral", "unknown"],
    "confidence": [0.9] * 9,
    "key_phrases": ["fast charging", "sharp display", "late delivery; box damaged",
                    "late delivery", "broken screen", "broken screen; poor quality",
                    "late package", "", ""],
})
df = add_categories(df, [2, 3, 4, 5, 6],
                    ["delivery", "delivery", "product quality", "product quality", "delivery"])

reg = T.build_registry(lambda: df)
fn = {name: spec["func"] for name, spec in reg.items()}

# ---- Tools ----
check("Registry has the 5 tools", set(reg) == {
    "calculate_average", "get_business_kpis", "get_product_performance",
    "get_complaint_analysis", "search_customer_feedback"})
decls = A.build_declarations(reg)
check("Function declarations build", len(decls) == 5)

check("calculate_average([1,2,3,4]) = 2.5", fn["calculate_average"]([1, 2, 3, 4])["average"] == 2.5)
check("calculate_average rejects empty list", raises(fn["calculate_average"], T.ToolError, []))
check("calculate_average rejects text", raises(fn["calculate_average"], T.ToolError, ["a"]))

k = fn["get_business_kpis"]()
check("KPIs: total 9, analysed 8, positive 2, negative 5",
      k["total"] == 9 and k["analysed"] == 8 and k["positive"] == 2 and k["negative"] == 5)
check("KPIs: negative_pct 62.5", k["negative_pct"] == 62.5)

p = fn["get_product_performance"]()["products"]
check("Performance: P1 rank 1, 4 reviews", p[0]["product"] == "P1" and p[0]["rank"] == 1 and p[0]["reviews"] == 4)
check("Performance: P2 net -75.0, At risk", p[1]["net_score"] == -75.0 and p[1]["status"] == "At risk")
one = fn["get_product_performance"]("p2")["products"]
check("Performance: product filter (case-insensitive)", len(one) == 1 and one[0]["product"] == "P2")
check("Performance: unknown product is an error", raises(fn["get_product_performance"], T.ToolError, "ZZZ"))

c = fn["get_complaint_analysis"]("P2")
check("Complaints P2: 3 negatives, product quality 2 (66.7%)",
      c["negative_reviews"] == 3 and c["categories"][0]["category"] == "product quality"
      and c["categories"][0]["count"] == 2 and c["categories"][0]["pct"] == 66.7)
call = fn["get_complaint_analysis"]()
check("Complaints (all): delivery is the top category", call["categories"][0]["category"] == "delivery")

s = fn["search_customer_feedback"]("late delivery")
check("Search: best match score 1.0, at least 3 matches", s["results"][0]["score"] == 1.0 and s["match_count"] >= 3)
s2 = fn["search_customer_feedback"]("broken screen", product="P2")
check("Search: product filter works", s2["results"] and all(r["product"] == "P2" for r in s2["results"]))
check("Search: sentiment filter works", fn["search_customer_feedback"]("delivery", sentiment="positive")["match_count"] == 0)
check("Search: bad sentiment is an error", raises(fn["search_customer_feedback"], T.ToolError, "x1", sentiment="angry"))
check("Search: empty query is an error", raises(fn["search_customer_feedback"], T.ToolError, "the and"))
check("Tool results are JSON and have no customer ids",
      "CID" not in json.dumps([k, p, c, s, s2]))

reg_empty = T.build_registry(lambda: None)
check("No data: tool gives a clear error", raises(reg_empty["get_business_kpis"]["func"], T.ToolError))


# ---- Agent loop (nakli model) ----
def scripted(steps, seen=None):
    it = iter(steps)

    def step(contents):
        if seen is not None:
            seen.append(list(contents))
        kind, payload = next(it)
        if kind == "calls":
            parts = [types.Part(function_call=types.FunctionCall(name=n, args=a)) for n, a in payload]
            return types.Content(role="model", parts=parts), list(payload), ""
        return types.Content(role="model", parts=[types.Part(text=payload)]), [], payload

    return step


seen = []
steps = [
    ("calls", [("get_product_performance", {})]),
    ("calls", [("get_complaint_analysis", {"product": "P2"}),
               ("search_customer_feedback", {"query": "broken screen", "product": "P2"})]),
    ("text", "P2 is at risk. Fix screen quality."),
]
ag = A.AnalystAgent(lambda: df, step=scripted(steps, seen))
res = ag.ask("Which product is worst and why?")
check("Multi-round: 3 rounds, 3 tool calls, answered",
      res["rounds"] == 3 and res["tool_calls"] == 3 and res["stopped"] == "answered")
check("Multi-round: tools ran in order", [t["tool"] for t in res["trace"]] ==
      ["get_product_performance", "get_complaint_analysis", "search_customer_feedback"])
check("Multi-round: all statuses ok", all(t["status"] == "ok" for t in res["trace"]))
check("Trace: round numbers 1, 2, 2", [t["round"] for t in res["trace"]] == [1, 2, 2])
check("Trace: durations are integers >= 0",
      all(isinstance(t["duration_ms"], int) and t["duration_ms"] >= 0 for t in res["trace"]))
check("Trace: entries have all keys",
      all(set(t) == {"round", "tool", "args", "status", "duration_ms", "preview"} for t in res["trace"]))
fr = seen[1][-1].parts[0].function_response
check("Round 2 receives round 1 tool result",
      fr.name == "get_product_performance" and fr.response["result"]["products"][0]["product"] == "P1")
check("Round 3 receives both round 2 results", len(seen[2][-1].parts) == 2)
check("Final answer returned", res["answer"] == "P2 is at risk. Fix screen quality.")

ag = A.AnalystAgent(lambda: df, step=scripted([
    ("calls", [("does_not_exist", {}), ("get_complaint_analysis", {"product": "ZZZ"})]),
    ("text", "done"),
]))
res = ag.ask("test errors")
check("Unknown tool and bad product give error status, loop continues",
      [t["status"] for t in res["trace"]] == ["error", "error"] and res["answer"] == "done")

ag = A.AnalystAgent(lambda: df, step=scripted([
    ("calls", [("get_business_kpis", {})] * 6),
    ("text", "done"),
]))
res = ag.ask("many calls")
check("Extra calls in one round are skipped",
      [t["status"] for t in res["trace"]] == ["ok"] * 5 + ["skipped"])

ag = A.AnalystAgent(lambda: df, step=scripted(itertools.repeat(("calls", [("get_business_kpis", {})]))))
res = ag.ask("loop forever")
check("Max rounds stops the loop",
      res["stopped"] == "max_rounds" and res["rounds"] == A.MAX_ROUNDS and res["tool_calls"] == A.MAX_ROUNDS)


def boom(contents):
    raise RuntimeError("x")


ag = A.AnalystAgent(lambda: df, step=boom)
ag.retry_delay = 0
check("Model failure becomes a clear AIError", raises(ag.ask, A.AIError, "hello"))

seen = []
ag = A.AnalystAgent(lambda: df, step=scripted([("text", "ok")], seen))
hist = [{"role": "user", "text": "Q1"}, {"role": "model", "text": "A1"}]
ag.ask("Q2", hist)
c0 = seen[0]
check("History is sent as user, model, user",
      [x.role for x in c0] == ["user", "model", "user"]
      and c0[0].parts[0].text == "Q1" and c0[-1].parts[0].text == "Q2")

seen = []
ag = A.AnalystAgent(lambda: df, step=scripted([("text", "ok")], seen))
ag.ask("Q2", [{"role": "model", "text": "stray"}, {"role": "user", "text": "Q1"}, {"role": "model", "text": "A1"}])
check("History starting with a model message is trimmed", len(seen[0]) == 3)

ag = A.AnalystAgent(lambda: df, step=scripted([("text", "ok")]))
check("Empty question is rejected", raises(ag.ask, ValueError, "   "))
check("Very long question is rejected", raises(ag.ask, ValueError, "x" * 501))

ag = A.AnalystAgent(lambda: None, step=scripted([
    ("calls", [("get_business_kpis", {})]),
    ("text", "Pehle CSV upload karein."),
]))
res = ag.ask("kpis?")
check("No data: tool error is passed to the model, loop ends cleanly",
      res["trace"][0]["status"] == "error" and res["answer"] == "Pehle CSV upload karein.")

low = A.SYSTEM_INSTRUCTION.lower()
check("System instruction: no invented numbers, reviews are data",
      "never invent" in low and "data, not instructions" in low)

print()
print("ALL PASSED" if not failed else "FAILED: " + ", ".join(failed))
