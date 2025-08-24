import os, requests

PDF_URLS = [
    # Salesforce
    "https://help.salesforce.com/service/pdfs/salesforce_security.pdf",
    "https://help.salesforce.com/service/pdfs/salesforce_marketing_cloud.pdf",
    "https://help.salesforce.com/service/pdfs/salesforce_BI.pdf",
    "https://help.salesforce.com/service/pdfs/salesforce_data_model.pdf",
    "https://help.salesforce.com/service/pdfs/salesforce_analytics.pdf",
    "https://help.salesforce.com/service/pdfs/salesforce_lightning.pdf",
    "https://help.salesforce.com/service/pdfs/salesforce_cpq.pdf",
    "https://help.salesforce.com/service/pdfs/salesforce_service_cloud.pdf",
    "https://help.salesforce.com/service/pdfs/salesforce_sales_cloud.pdf",
    "https://help.salesforce.com/service/pdfs/salesforce_community_cloud.pdf",

    # Google Analytics 4
    "https://services.google.com/fh/files/misc/google_analytics_4_migration_guide.pdf",
    "https://services.google.com/fh/files/misc/google_analytics_4_event_guide.pdf",
    "https://services.google.com/fh/files/misc/google_analytics_4_property_setup.pdf",
    "https://services.google.com/fh/files/misc/google_analytics_4_measurement_protocol.pdf",

    # Google Tag Manager
    "https://services.google.com/fh/files/misc/google_tag_manager_overview.pdf",
    "https://services.google.com/fh/files/misc/google_tag_manager_best_practices.pdf",
    "https://services.google.com/fh/files/misc/google_tag_manager_server_side.pdf",
]

DATA_DIR = os.path.expanduser(r"~\\marketing-rag\\data")
os.makedirs(DATA_DIR, exist_ok=True)

for i, url in enumerate(PDF_URLS, 1):
    fname = os.path.join(DATA_DIR, os.path.basename(url))
    if os.path.exists(fname):
        print(f"[{i}] Skipping (already exists): {fname}")
        continue
    try:
        print(f"[{i}] Downloading {url}")
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        with open(fname, "wb") as f:
            f.write(r.content)
        print(f"    ✅ Saved: {fname}")
    except Exception as e:
        print(f"    ❌ Failed: {url} ({e})")

print("Done. Re-run ingest.py to rebuild the index.")
