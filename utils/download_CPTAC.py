from idc_index import index
client = index.IDCClient()

# 먼저 dry_run으로 대상/용량 확인
client.download_from_selection(
    collection_id="cptac_pda",
    downloadDir="./data/cptac_pda_wsi",
    dry_run=True
)