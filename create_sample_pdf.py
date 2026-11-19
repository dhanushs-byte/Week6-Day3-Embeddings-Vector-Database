from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER

OUTPUT = "data/sample.pdf"

sections = [
    ("Introduction", "This document explains a practical support knowledge base. It describes how information is organized, searched, updated, and maintained. A clear structure helps readers find the correct answer quickly."),
    ("Account Access", "Users can sign in with their registered email address and password. If a password is forgotten, the user should select the password reset option and follow the instructions sent to the registered email address. Never share a password or one-time code."),
    ("Password Reset", "A password reset link is sent to the registered email address. The link is valid for a limited period. If the message is not received, check the spam folder and confirm that the correct email address was entered."),
    ("Billing and Invoices", "Monthly invoices are available in the billing section of the account. Users can download a PDF copy for their records. The invoice includes the billing period, invoice number, amount, and payment status."),
    ("Payment Methods", "Users can update an expired payment method from account settings. The new payment method is used for the next eligible billing cycle. Payment details should only be entered on the secure payment page."),
    ("Subscription Management", "A subscription can be reviewed from the subscription page. The page shows the current plan, renewal date, and available plan options. Changes may take effect immediately or at the next renewal depending on the selected option."),
    ("Data Privacy", "Personal information should be collected only when it is needed to provide the service. Access to customer information must be limited to authorized staff. Sensitive information must not be included in public support messages."),
    ("Troubleshooting", "When a page does not load, refresh the browser and check the network connection. If the issue continues, clear the browser cache and try again. Include the error message when contacting support."),
    ("Customer Support", "Customers can contact support through the help centre. A useful support request includes a short description, the steps to reproduce the issue, the approximate time, and relevant screenshots without exposing passwords or payment details."),
    ("Document Maintenance", "The knowledge base should be reviewed regularly. When a policy changes, update the affected section and record the revision date. Stable identifiers help update existing chunks instead of creating duplicate records."),
]

styles = getSampleStyleSheet()
title_style = ParagraphStyle("PageTitle", parent=styles["Heading1"], fontSize=18, leading=22, spaceAfter=14)
body_style = ParagraphStyle("BodyCustom", parent=styles["BodyText"], fontSize=11, leading=17, spaceAfter=10)
doc = SimpleDocTemplate(OUTPUT, pagesize=A4, rightMargin=55, leftMargin=55, topMargin=55, bottomMargin=55)
story = []
for i, (heading, body) in enumerate(sections, 1):
    story.append(Paragraph(f"Knowledge Base Guide — Page {i}", styles["Title"]))
    story.append(Spacer(1, 18))
    story.append(Paragraph(heading, title_style))
    story.append(Paragraph(body, body_style))
    story.append(Paragraph("Related guidance: use the relevant account page, follow the displayed instructions, and contact support if the issue remains unresolved.", body_style))
    if i != len(sections):
        story.append(PageBreak())
doc.build(story)
print(f"Created {OUTPUT}")
