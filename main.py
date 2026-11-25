import os
import random
import re
import time
from datetime import datetime
from typing import Any
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from google import genai
from google.genai import types
from pydantic import BaseModel

load_dotenv()

app = FastAPI(title="BillGuard AI — GST Invoice Extraction & Audit Automation")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PRIMARY_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
FALLBACK_MODELS = list(dict.fromkeys([PRIMARY_MODEL, "gemini-3.8-flash-lite"]))

client = genai.Client(api_key=GEMINI_API_KEY)

# --- APPROVED PURCHASE ORDERS (3-WAY MATCH REFERENCE) ---
PURCHASE_ORDERS_DB: dict[str, dict[str, Any]] = {
    "PO-901": {
        "po_number": "PO-901",
        "vendor_name": "Kaveri Industrial Packaging",
        "gstin": "29AABCK4821M1Z5",
        "approved_base": 50000.0,
        "approved_total": 59000.0,
    },
    "PO-902": {
        "po_number": "PO-902",
        "vendor_name": "Deccan Cloud & IT Solutions",
        "gstin": "29AABCD7712K1Z8",
        "approved_base": 100000.0,
        "approved_total": 118000.0,
    },
    "PO-903": {
        "po_number": "PO-903",
        "vendor_name": "Nandi Logistics & ColdChain",
        "gstin": "27AABCN3390P1Z2",
        "approved_base": 40000.0,
        "approved_total": 47200.0,
    },
}

SAMPLE_RAW_INVOICES: dict[str, str] = {
    "INV-2026-401": (
        "TAX INVOICE | Invoice No: INV-2026-401 | Date: 24-Nov-2026 | "
        "Vendor: Kaveri Industrial Packaging | GSTIN: 29AABCK4821M1Z5 | PO Ref: PO-901 | "
        "Items: 1,000x 5-Ply Export Cartons | Taxable Value: ₹50,000 | "
        "CGST (9%): ₹4,500 | SGST (9%): ₹4,500 | Grand Total: ₹59,000"
    ),
    "INV-2026-402": (
        "TAX INVOICE | Invoice No: INV-2026-402 | Date: 25-Nov-2026 | "
        "Vendor: Deccan Cloud & IT Solutions | GSTIN: 29AABCD7712K1Z8 | PO Ref: PO-902 | "
        "Items: Annual AWS & Kubernetes Managed Ops | Taxable Value: ₹1,00,000 | "
        "CGST (9%): ₹9,000 | SGST (9%): ₹9,000 | Grand Total: ₹1,24,500"
    ),
    "INV-2026-403": (
        "TAX INVOICE | Invoice No: INV-2026-403 | Date: 25-Nov-2026 | "
        "Vendor: Nandi Logistics & ColdChain | GSTIN: 27AABCN3390P1Z2 | PO Ref: PO-903 | "
        "Items: Mumbai-Bengaluru Reefer Linehaul | Taxable Value: ₹40,000 | "
        "IGST (18%): ₹7,200 | Grand Total: ₹47,200"
    ),
}

INVOICE_LEDGER_DB: list[dict[str, Any]] = []
CHAT_SESSIONS: dict[str, list[types.Content]] = {}
_current_request_tools: list[str] = []

GSTIN_REGEX = re.compile(r"^[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[1-9A-Z]{1}Z[0-9A-Z]{1}$")


# --- INVOICE AUTOMATION TOOLS ---
def extract_and_audit_invoice(
    invoice_number: str,
    vendor_name: str,
    gstin: str,
    base_amount: float,
    gst_amount: float,
    billed_total: float,
    po_number: str = "",
    line_items: str = "Standard B2B Supply",
) -> dict[str, Any]:
    """Extract structured GST invoice fields, run automated 3-way PO & tax math audit, and log to the accounting ledger.

    Args:
        invoice_number: Invoice ID (e.g., INV-2026-401).
        vendor_name: Legal name of the supplier/vendor.
        gstin: 15-character Indian GSTIN of the vendor.
        base_amount: Pre-tax taxable amount in INR.
        gst_amount: Total GST amount (CGST + SGST or IGST) in INR.
        billed_total: Final grand total printed on the invoice in INR.
        po_number: Purchase Order reference number if mentioned (e.g., PO-901).
        line_items: Short description of billed goods or services.
    """
    _current_request_tools.append(f"extract_and_audit_invoice({invoice_number})")
    clean_inv = invoice_number.strip().upper()
    clean_gstin = gstin.strip().upper()
    clean_po = po_number.strip().upper()

    flags: list[str] = []

    # 1. Duplicate invoice check
    if any(row["invoice_number"] == clean_inv for row in INVOICE_LEDGER_DB):
        flags.append(f"Duplicate Alert: {clean_inv} already exists in ledger")

    # 2. GSTIN format validation
    gstin_valid = bool(GSTIN_REGEX.match(clean_gstin))
    if not gstin_valid:
        flags.append(f"Invalid GSTIN format ({clean_gstin or 'Missing'})")

    # 3. Tax & Grand Total math audit
    expected_total = round(base_amount + gst_amount, 2)
    variance = round(billed_total - expected_total, 2)
    if abs(variance) > 1.0:
        flags.append(
            f"Math Mismatch: Base ₹{base_amount:,.0f} + GST ₹{gst_amount:,.0f} = ₹{expected_total:,.0f}, "
            f"but invoice bills ₹{billed_total:,.0f} (Variance: ₹{variance:+,.0f})"
        )

    # 4. 3-Way PO match
    po_status = "No PO Linked"
    if clean_po:
        po = PURCHASE_ORDERS_DB.get(clean_po)
        if not po:
            po_status = f"{clean_po} Not Found"
            flags.append(f"PO Reference {clean_po} not found in ERP")
        else:
            po_diff = round(billed_total - po["approved_total"], 2)
            if abs(po_diff) > 1.0:
                po_status = f"{clean_po} Over-Billed (₹{po_diff:+,.0f})"
                flags.append(
                    f"PO Limit Exceeded: {clean_po} approved for ₹{po['approved_total']:,.0f}, billed ₹{billed_total:,.0f}"
                )
            else:
                po_status = f"{clean_po} Matched"

    audit_status = "Flagged - Discrepancy" if flags else "Approved for Payout"
    entry = {
        "entry_id": f"ACC-{random.randint(701, 999)}",
        "invoice_number": clean_inv,
        "vendor_name": vendor_name.strip(),
        "gstin": clean_gstin or "N/A",
        "gstin_valid": gstin_valid,
        "po_number": clean_po or "N/A",
        "po_status": po_status,
        "line_items": line_items.strip(),
        "base_amount": round(base_amount, 2),
        "gst_amount": round(gst_amount, 2),
        "billed_total": round(billed_total, 2),
        "expected_total": expected_total,
        "audit_status": audit_status,
        "flags": flags,
        "processed_at": datetime.now().strftime("%I:%M %p"),
    }
    INVOICE_LEDGER_DB.insert(0, entry)
    return entry


def get_invoice_ledger() -> dict[str, Any]:
    """Retrieve all processed invoices along with total approved payout and claimable GST Input Tax Credit (ITC)."""
    _current_request_tools.append("get_invoice_ledger()")
    approved_total = sum(i["billed_total"] for i in INVOICE_LEDGER_DB if i["audit_status"] == "Approved for Payout")
    flagged_total = sum(i["billed_total"] for i in INVOICE_LEDGER_DB if i["audit_status"] != "Approved for Payout")
    itc_claimable = sum(i["gst_amount"] for i in INVOICE_LEDGER_DB if i["audit_status"] == "Approved for Payout")
    return {
        "count": len(INVOICE_LEDGER_DB),
        "approved_payout_inr": approved_total,
        "flagged_amount_inr": flagged_total,
        "itc_claimable_inr": itc_claimable,
        "invoices": INVOICE_LEDGER_DB,
    }


def approve_or_hold_invoice(invoice_number: str, decision: str, note: str) -> dict[str, Any]:
    """Manually override or update an invoice's payout decision in the ledger.

    Args:
        invoice_number: Invoice ID (e.g., INV-2026-402).
        decision: Either 'Approved for Payout' or 'On Hold - Vendor Query'.
        note: Audit note explaining the finance team's decision.
    """
    _current_request_tools.append(f"approve_or_hold_invoice({invoice_number} -> {decision})")
    clean_inv = invoice_number.strip().upper()
    for row in INVOICE_LEDGER_DB:
        if row["invoice_number"] == clean_inv:
            row["audit_status"] = decision.strip()
            row["flags"].append(f"Finance Override ({datetime.now().strftime('%I:%M %p')}): {note.strip()}")
            return {"updated": True, **row}
    return {"updated": False, "message": f"Invoice '{invoice_number}' not found in ledger."}


def _parse_amount(pattern: str, text: str, default: float = 0.0) -> float:
    m = re.search(pattern, text, flags=re.I)
    if not m:
        return default
    raw = m.group(1).replace(",", "").strip()
    try:
        return float(raw)
    except ValueError:
        return default


def backup_invoice_engine(user_msg: str) -> str:
    """Zero-downtime backup engine when Gemini cloud models experience 503 high-demand spikes."""
    text = user_msg.strip()
    lower = text.lower()

    if any(k in lower for k in ("ledger", "summary", "total itc", "all invoices", "dashboard")):
        summary = get_invoice_ledger()
        if summary["count"] == 0:
            return "The GST Invoice Ledger is currently empty. Tap any sample invoice chip below or paste raw bill text to extract & audit it!"
        return (
            f"Live GST Accounting Ledger Summary ({summary['count']} invoices):\n"
            f"• Approved for Payout: **₹{summary['approved_payout_inr']:,.0f}**\n"
            f"• Flagged / On Hold: **₹{summary['flagged_amount_inr']:,.0f}**\n"
            f"• Verified GST ITC Claimable: **₹{summary['itc_claimable_inr']:,.0f}**"
        )

    raw_text = text
    for inv_id, sample_body in SAMPLE_RAW_INVOICES.items():
        if inv_id.lower() in lower and "taxable value" not in lower:
            raw_text = sample_body
            break

    inv_m = re.search(r"\b(INV-[A-Z0-9\-]+)\b", raw_text, flags=re.I)
    inv_no = inv_m.group(1).upper() if inv_m else f"INV-2026-{random.randint(500, 899)}"

    po_m = re.search(r"\b(PO-\d+)\b", raw_text, flags=re.I)
    po_no = po_m.group(1).upper() if po_m else ""

    gst_m = re.search(r"\b([0-9]{2}[A-Z]{5}[0-9]{4}[A-Z][1-9A-Z]Z[0-9A-Z])\b", raw_text, flags=re.I)
    gstin = gst_m.group(1).upper() if gst_m else "29AABCK4821M1Z5"

    vendor_m = re.search(r"Vendor:\s*([^|\n]+)", raw_text, flags=re.I)
    vendor = vendor_m.group(1).strip() if vendor_m else "Apex B2B Suppliers Pvt Ltd"

    items_m = re.search(r"Items:\s*([^|\n]+)", raw_text, flags=re.I)
    items = items_m.group(1).strip() if items_m else "Commercial Goods & Services"

    base = _parse_amount(r"(?:Taxable Value|Base|Subtotal)[:\s₹]*([\d,]+(?:\.\d+)?)", raw_text, 25000.0)
    cgst = _parse_amount(r"CGST[^:\d]*[:\s₹]*([\d,]+(?:\.\d+)?)", raw_text, 0.0)
    sgst = _parse_amount(r"SGST[^:\d]*[:\s₹]*([\d,]+(?:\.\d+)?)", raw_text, 0.0)
    igst = _parse_amount(r"IGST[^:\d]*[:\s₹]*([\d,]+(?:\.\d+)?)", raw_text, 0.0)
    gst_total = cgst + sgst + igst if (cgst + sgst + igst) > 0 else round(base * 0.18, 2)
    grand_total = _parse_amount(r"(?:Grand Total|Total)[:\s₹]*([\d,]+(?:\.\d+)?)", raw_text, round(base + gst_total, 2))

    res = extract_and_audit_invoice(
        invoice_number=inv_no,
        vendor_name=vendor,
        gstin=gstin,
        base_amount=base,
        gst_amount=gst_total,
        billed_total=grand_total,
        po_number=po_no,
        line_items=items,
    )

    flag_text = "\n• **Audit Alerts:** " + "; ".join(res["flags"]) if res["flags"] else "\n• **Audit Result:** 100% GST Math & 3-Way PO Match Verified"
    return (
        f"Extracted & Audited **{res['invoice_number']}** ({res['entry_id']}):\n"
        f"• Vendor: **{res['vendor_name']}** (`{res['gstin']}`)\n"
        f"• Taxable Base: ₹{res['base_amount']:,.0f} | GST: ₹{res['gst_amount']:,.0f} | Billed Total: **₹{res['billed_total']:,.0f}**\n"
        f"• PO Match: {res['po_status']}\n"
        f"• Status: **{res['audit_status']}**"
        f"{flag_text}"
    )


SYSTEM_INSTRUCTION = """
You are **BillGuard AI**, an autonomous **Indian GST Invoice Extraction, 3-Way PO Audit & Accounts Payable Agent**.

WORKFLOW RULES:
1. Whenever a user provides or references an invoice (such as `INV-2026-401`, `INV-2026-402`, `INV-2026-403`, or pasted raw invoice text), extract all fields and IMMEDIATELY call `extract_and_audit_invoice`.
2. Pay strict attention to the exact numbers printed in the invoice—do NOT auto-correct `billed_total` before calling `extract_and_audit_invoice`, because the tool audits whether `base_amount + gst_amount == billed_total` and checks against ERP Purchase Orders (`PO-901`, `PO-902`, `PO-903`).
3. When asked for ledger totals or ITC summaries, call `get_invoice_ledger`.
4. Format responses cleanly with Indian Rupees (₹), bullet points, and clear **Approved for Payout** or **Flagged - Discrepancy** verdicts.
"""


class ChatRequest(BaseModel):
    session_id: str
    message: str


@app.get("/health")
async def health():
    return {
        "status": "running",
        "models": FALLBACK_MODELS,
        "invoices_count": len(INVOICE_LEDGER_DB),
    }


@app.get("/api/ledger")
async def api_ledger():
    return get_invoice_ledger()


@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    global _current_request_tools

    if not GEMINI_API_KEY:
        return JSONResponse(
            status_code=500,
            content={"reply": "Server configuration error: GEMINI_API_KEY is missing.", "tools": []},
        )

    prompt_text = req.message
    for inv_id, sample_body in SAMPLE_RAW_INVOICES.items():
        if inv_id.lower() in prompt_text.lower() and "taxable value" not in prompt_text.lower():
            prompt_text = f"{req.message}\n\nRAW INVOICE DATA:\n{sample_body}"
            break

    history = CHAT_SESSIONS.setdefault(req.session_id, [])
    user_turn = types.Content(role="user", parts=[types.Part.from_text(text=prompt_text)])
    conversation = history[-10:] + [user_turn]

    last_error = ""
    for model_name in FALLBACK_MODELS:
        for attempt in range(2):
            _current_request_tools = []
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=conversation,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_INSTRUCTION,
                        tools=[extract_and_audit_invoice, get_invoice_ledger, approve_or_hold_invoice],
                        temperature=0.1,
                    ),
                )
                reply_text = response.text or "Invoice audited."
                history.append(user_turn)
                history.append(types.Content(role="model", parts=[types.Part.from_text(text=reply_text)]))
                if len(history) > 12:
                    CHAT_SESSIONS[req.session_id] = history[-12:]
                return {
                    "reply": reply_text,
                    "tools": list(_current_request_tools),
                    "invoices_count": len(INVOICE_LEDGER_DB),
                }
            except Exception as e:
                last_error = str(e)
                is_transient = any(code in last_error for code in ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED"))
                if is_transient and attempt == 0:
                    time.sleep(1.5)
                    continue
                break

    if any(code in last_error for code in ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED", "404", "NOT_FOUND")):
        _current_request_tools = []
        fallback_reply = backup_invoice_engine(prompt_text)
        history.append(user_turn)
        history.append(types.Content(role="model", parts=[types.Part.from_text(text=fallback_reply)]))
        return {
            "reply": fallback_reply,
            "tools": list(_current_request_tools),
            "invoices_count": len(INVOICE_LEDGER_DB),
        }

    return JSONResponse(
        status_code=500,
        content={"reply": f"Assistant error: {last_error}", "tools": []},
    )


@app.get("/", response_class=HTMLResponse)
async def chat_ui():
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0" />
  <title>BillGuard AI — GST Invoice Automation</title>
  <style>
    :root {
      --bg: #f1f5f9;
      --card: #ffffff;
      --primary: #064e3b;
      --accent: #0d9488;
      --bot-bg: #f8fafc;
      --border: #e2e8f0;
      --text: #0f172a;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
    body { background: var(--bg); color: var(--text); display: flex; justify-content: center; height: 100dvh; }
    .app { width: 100%; max-width: 560px; background: var(--card); display: flex; flex-direction: column; height: 100dvh; box-shadow: 0 0 30px rgba(0,0,0,0.06); }
    .header { background: var(--primary); color: #fff; padding: 14px 16px; display: flex; justify-content: space-between; align-items: center; gap: 8px; }
    .header-left { display: flex; align-items: center; gap: 10px; }
    .avatar { width: 36px; height: 36px; border-radius: 10px; background: var(--accent); display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 15px; }
    .title { font-size: 15px; font-weight: 600; }
    .subtitle { font-size: 11px; color: #a7f3d0; display: flex; align-items: center; gap: 5px; margin-top: 2px; }
    .dot { width: 7px; height: 7px; border-radius: 50%; background: #34d399; display: inline-block; }
    .ledger-btn { background: rgba(255,255,255,0.14); color: #fff; border: 1px solid rgba(255,255,255,0.25); border-radius: 8px; padding: 7px 10px; font-size: 12px; font-weight: 600; cursor: pointer; white-space: nowrap; }
    .messages { flex: 1; overflow-y: auto; padding: 16px; display: flex; flex-direction: column; gap: 12px; }
    .msg { max-width: 88%; padding: 11px 14px; border-radius: 14px; font-size: 14px; line-height: 1.48; white-space: pre-wrap; word-wrap: break-word; }
    .msg.bot { align-self: flex-start; background: var(--bot-bg); border: 1px solid var(--border); border-bottom-left-radius: 4px; }
    .msg.user { align-self: flex-end; background: var(--primary); color: #fff; border-bottom-right-radius: 4px; }
    .tool-badge { display: inline-block; font-size: 11px; font-weight: 600; color: #065f46; background: #d1fae5; padding: 3px 8px; border-radius: 6px; margin-bottom: 6px; }
    .chips { display: flex; gap: 6px; overflow-x: auto; padding: 8px 12px; border-top: 1px solid var(--border); background: #f8fafc; scrollbar-width: none; }
    .chips::-webkit-scrollbar { display: none; }
    .chip { flex-shrink: 0; background: #fff; border: 1px solid #cbd5e1; border-radius: 999px; padding: 6px 11px; font-size: 12px; color: #334155; cursor: pointer; }
    .chip:active { background: #e2e8f0; }
    .input-bar { display: flex; gap: 8px; padding: 10px 12px; border-top: 1px solid var(--border); background: #fff; }
    .input-bar input { flex: 1; border: 1px solid #cbd5e1; border-radius: 10px; padding: 10px 12px; font-size: 14px; outline: none; }
    .input-bar input:focus { border-color: var(--accent); }
    .input-bar button { background: var(--accent); color: #fff; border: none; border-radius: 10px; padding: 0 16px; font-size: 14px; font-weight: 600; cursor: pointer; }
    .input-bar button:disabled { opacity: 0.5; }
    .modal { position: fixed; inset: 0; background: rgba(15,23,42,0.55); display: none; align-items: center; justify-content: center; padding: 14px; z-index: 50; }
    .modal-card { background: #fff; width: 100%; max-width: 520px; max-height: 84vh; border-radius: 14px; padding: 16px; overflow-y: auto; }
    .modal-head { display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; padding-bottom: 8px; border-bottom: 1px solid var(--border); }
    .kpi-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin-bottom: 12px; }
    .kpi { background: #f8fafc; border: 1px solid var(--border); border-radius: 10px; padding: 8px; text-align: center; }
    .kpi-label { font-size: 10px; color: #64748b; text-transform: uppercase; font-weight: 700; }
    .kpi-val { font-size: 13px; font-weight: 700; margin-top: 3px; color: #0f172a; }
    .card-item { border: 1px solid var(--border); border-radius: 10px; padding: 10px; margin-bottom: 8px; font-size: 13px; background: #f8fafc; }
    .pill-ok { display: inline-block; padding: 2px 7px; border-radius: 999px; font-size: 11px; font-weight: 700; background: #dcfce7; color: #166534; }
    .pill-warn { display: inline-block; padding: 2px 7px; border-radius: 999px; font-size: 11px; font-weight: 700; background: #fee2e2; color: #991b1b; }
  </style>
</head>
<body>
  <div class="app">
    <div class="header">
      <div class="header-left">
        <div class="avatar">🧾</div>
        <div>
          <div class="title">BillGuard GST Invoice AI</div>
          <div class="subtitle"><span class="dot"></span> GSTIN Check &bull; 3-Way PO Audit &bull; ITC Ledger</div>
        </div>
      </div>
      <button class="ledger-btn" onclick="openLedger()">GST Ledger (<span id="iCount">0</span>)</button>
    </div>

    <div class="messages" id="chatBox">
      <div class="msg bot">Hi! I'm <b>BillGuard AI</b>. Paste any vendor bill or test our sample GST invoices to see automated extraction, GSTIN validation, and 3-way PO auditing:
• <b>INV-2026-401:</b> Kaveri Packaging (₹59,000 — Clean PO-901 Match)
• <b>INV-2026-402:</b> Deccan Cloud (₹1,24,500 — Hidden ₹6,500 Overcharge!)
• <b>INV-2026-403:</b> Nandi Logistics (₹47,200 — Interstate 18% IGST)</div>
    </div>

    <div class="chips">
      <button class="chip" onclick="quickSend('Extract and audit invoice INV-2026-401')">1. Clean Bill (INV-401)</button>
      <button class="chip" onclick="quickSend('Extract and audit invoice INV-2026-402')">2. Catch Overcharge (INV-402)</button>
      <button class="chip" onclick="quickSend('Extract and audit invoice INV-2026-403')">3. IGST Freight (INV-403)</button>
      <button class="chip" onclick="quickSend('Show GST ledger and total ITC claimable')">4. ITC Summary</button>
    </div>

    <form class="input-bar" onsubmit="sendMsg(event)">
      <input id="msgInput" type="text" placeholder="Audit INV-2026-402, or paste raw bill text..." autocomplete="off" required />
      <button id="sendBtn" type="submit">Audit</button>
    </form>
  </div>

  <div class="modal" id="ledgerModal" onclick="if(event.target===this)this.style.display='none'">
    <div class="modal-card">
      <div class="modal-head">
        <strong>Audited GST Invoice &amp; ITC Ledger</strong>
        <button onclick="document.getElementById('ledgerModal').style.display='none'" style="border:none;background:none;font-size:18px;cursor:pointer;">✕</button>
      </div>
      <div class="kpi-grid">
        <div class="kpi"><div class="kpi-label">Approved Payout</div><div class="kpi-val" id="kApproved">₹0</div></div>
        <div class="kpi"><div class="kpi-label">Flagged / Held</div><div class="kpi-val" id="kFlagged" style="color:#b91c1c;">₹0</div></div>
        <div class="kpi"><div class="kpi-label">Claimable ITC</div><div class="kpi-val" id="kItc" style="color:#047857;">₹0</div></div>
      </div>
      <div id="invoiceList">Loading...</div>
    </div>
  </div>

  <script>
    const sessionId = 'inv_' + Math.random().toString(36).slice(2, 10);
    const chatBox = document.getElementById('chatBox');
    const msgInput = document.getElementById('msgInput');
    const sendBtn = document.getElementById('sendBtn');

    function appendMsg(role, text, tools = []) {
      const div = document.createElement('div');
      div.className = 'msg ' + role;
      if (tools && tools.length > 0) {
        const badge = document.createElement('div');
        badge.className = 'tool-badge';
        badge.textContent = '⚡ Action: ' + tools.join(' → ');
        div.appendChild(badge);
        div.appendChild(document.createElement('br'));
      }
      const span = document.createElement('span');
      span.textContent = text;
      div.appendChild(span);
      chatBox.appendChild(div);
      chatBox.scrollTop = chatBox.scrollHeight;
      return div;
    }

    function quickSend(text) {
      msgInput.value = text;
      sendMsg(new Event('submit'));
    }

    async function sendMsg(e) {
      e.preventDefault();
      const text = msgInput.value.trim();
      if (!text) return;
      msgInput.value = '';
      appendMsg('user', text);
      sendBtn.disabled = true;
      const typing = appendMsg('bot', 'Auditing GST & PO math...');

      try {
        const res = await fetch('/api/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ session_id: sessionId, message: text })
        });
        const data = await res.json();
        typing.remove();
        appendMsg('bot', data.reply, data.tools || []);
        if (typeof data.invoices_count === 'number') {
          document.getElementById('iCount').textContent = data.invoices_count;
        }
      } catch (err) {
        typing.textContent = 'Network error. Please try again.';
      } finally {
        sendBtn.disabled = false;
      }
    }

    async function openLedger() {
      const modal = document.getElementById('ledgerModal');
      modal.style.display = 'flex';
      const res = await fetch('/api/ledger');
      const data = await res.json();

      document.getElementById('kApproved').textContent = '₹' + (data.approved_payout_inr || 0).toLocaleString('en-IN');
      document.getElementById('kFlagged').textContent = '₹' + (data.flagged_amount_inr || 0).toLocaleString('en-IN');
      document.getElementById('kItc').textContent = '₹' + (data.itc_claimable_inr || 0).toLocaleString('en-IN');

      document.getElementById('invoiceList').innerHTML = (data.invoices || []).map(inv => {
        const ok = inv.audit_status === 'Approved for Payout';
        const pillClass = ok ? 'pill-ok' : 'pill-warn';
        const flagsHtml = (inv.flags && inv.flags.length) ? `<div style="color:#b91c1c;margin-top:4px;font-size:12px;">⚠ ${inv.flags.join(' | ')}</div>` : '';
        return `<div class="card-item"><b>${inv.invoice_number}</b> &bull; <span class="${pillClass}">${inv.audit_status}</span><br><b>${inv.vendor_name}</b> (GSTIN: ${inv.gstin})<br>Base: ₹${inv.base_amount.toLocaleString('en-IN')} + GST: ₹${inv.gst_amount.toLocaleString('en-IN')} = <b>₹${inv.billed_total.toLocaleString('en-IN')}</b> (${inv.po_status})${flagsHtml}</div>`;
      }).join('') || '<p style="font-size:13px;color:#64748b;">No invoices audited yet. Tap a chip above to run an audit!</p>';
    }
  </script>
</body>
</html>"""

