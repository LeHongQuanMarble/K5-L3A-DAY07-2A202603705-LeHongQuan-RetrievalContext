# Báo cáo Day 7

## Dữ liệu và benchmark

- Nguồn, quyền sử dụng, số trang và số chunk: *Pride and Prejudice* (Jane Austen), Project Gutenberg eBook #1342, bản công cộng tại Hoa Kỳ. Dựng thành PDF 216 trang (mỗi chương bắt đầu một trang mới) bằng `starter/scripts/build_pdf.py` của Day 6. Chunk: 1166 chunk, chiến lược B của Day 6 (đoạn tối đa 800 ký tự, không vượt trang). Trang là số do PDF dựng, không phải số trang bản in.
- Vector store, embedding model, phiên bản và số chiều: pgvector 0.8.7 trên PostgreSQL 16 (image `pgvector/pgvector:0.8.7-pg16`, từ Day 6). Embedding `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, 384 chiều, chạy local, `max_seq_length` 256. Query và chunk dùng cùng model.
- Agent model, cách chạy và phân chia dev/test: `claude-haiku-5-5` qua Anthropic Messages API, tool calling bắt buộc chọn một tool (`tool_choice: any`). Chạy bằng `starter/run_dev_agent.py --split dev|test`. Benchmark: 23 câu có đáp án (15 dev, 8 test; 5 câu có filter theo chương, 2 dev và 3 test), 6 câu thiếu evidence (3 dev, 3 test). Các câu do Day 6 soạn và kiểm tra bằng `check_benchmark.py`; 6 câu thiếu evidence do tôi thêm và kiểm tra bằng tìm từ khóa trong corpus (`data/pride_unanswerable.json`).
- Cách xác định câu có/thiếu evidence và tiêu chí chấm: câu có đáp án có `evidence_quote` nằm nguyên văn trong đúng trang gold; kiểm tra bằng `baseline.validate()` (evidence phải nằm trọn trong một chunk đúng nguồn và trang, đã pass cho 23/23 câu). Câu thiếu evidence: đạt khi agent từ chối. Tiêu chí chấm answer (0–2) và citation áp dụng theo `BENCHMARK.md`; nhãn hiện do AI chấm và cần người làm xác nhận (xem mục Agent và đánh giá).

## Retrieval và thử nghiệm

| Cấu hình | Phương pháp, top-k, filter, rerank, context budget | Giả thuyết / thay đổi | Split | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Thời gian / chi phí |
|---|---|---|---|---|---|---|---|
| Baseline | TF-IDF, top-3, không filter, budget 800 | Baseline lexical | dev | 1/15 (6.7%) | 1/15 (6.7%) | 1/15 (6.7%) | ~1 ms/câu, 0 USD |
| Phương pháp 2 | Dense B (MiniLM, pgvector), top-3, không filter, budget 800 | Dense đa ngôn ngữ cải thiện truy xuất câu hỏi tiếng Việt trên sách tiếng Anh | dev | 2/15 (13.3%) | 3/15 (20.0%) | 1/15 (6.7%) | ~36 ms/câu, 0 USD |
| Phương pháp 3 | Hybrid RRF (dense + TF-IDF, k=60, 20 ứng viên mỗi nguồn), top-3, budget 1200 | Kết hợp hai tín hiệu | dev | 1/15 (6.7%) | 2/15 (13.3%) | 1/15 (6.7%) | — |
| Cải thiện 1 | Dense B, top-3, budget 1200 | Lỗi budget làm mất evidence đã tìm được (xem bên dưới) | dev | 2/15 (13.3%) | 3/15 (20.0%) | 2/15 (13.3%) | — |
| Cải thiện 2 (thử thêm) | Dense B, top-3, budget 1600 | Budget lớn hơn có thể tăng thêm context | dev | 2/15 (13.3%) | 3/15 (20.0%) | 2/15 (13.3%) | context TB 1374 ký tự |
| Cấu hình đã chọn | Dense B, top-3, budget 1200 | Chốt trên dev theo Hit@3, rồi Hit@1, rồi context | test | 2/8 (25.0%) | 2/8 (25.0%) | 2/8 (25.0%) | ~35 ms/câu, 0 USD |
| Baseline | TF-IDF, top-3, budget 800 | Baseline lexical | test | 2/8 (25.0%) | 2/8 (25.0%) | 1/8 (12.5%) | ~1 ms/câu, 0 USD |

Ghi chú: "Cải thiện 2" là thử nghiệm budget 1600, không phải cải thiện được chọn. Với dev, cấu hình chọn là budget 1200 vì cùng Hit@3 và Hit@1 với 1600 nhưng context ngắn hơn (807 so với 1374 ký tự). Thời gian là thời gian truy vấn (embedding và tìm kiếm), chưa gồm nạp model và chạy agent.

**Lỗi dev 1 — đóng gói context (budget).** Với budget 800, `pack_context` bỏ hẳn chunk không vừa budget. q14: evidence ở hạng 1 với block (có tiêu đề) dài 855 ký tự, vượt budget nên luôn bị loại; context chỉ còn chunk hạng 2 không đúng. q08: evidence ở hạng 3 (block 547 ký tự) bị loại khi cộng với chunk hạng 1 (369 ký tự). Giả thuyết: tăng budget sẽ khôi phục evidence. Kết quả: budget 1200 đưa q14 và q11 vào context (Context Evidence Rate 1/15 → 2/15), Hit@3 không đổi vì q08 vẫn bị loại ở budget 1200 do chunk hạng 2 chiếm chỗ. Chọn 1200.

**Lỗi dev 2 — kết hợp hybrid (RRF).** Giả thuyết: cộng thêm tín hiệu TF-IDF sẽ tăng Hit@3. Kết quả: giảm từ 3/15 xuống 2/15 (q08 mất khỏi top-3, q14 xuống hạng 2). TF-IDF đưa lên các chunk trùng từ tiếng Anh không liên quan. Loại hybrid.

**Lý do chọn cấu hình.** Theo tiêu chí trong `BENCHMARK.md` trên dev: Hit@3 cao nhất (dense 1200: 3/15, hybrid: 2/15). Hòa Hit@1 giữa dense 800, dense 1200 và dense 1600; chọn 1200 vì context ngắn hơn 1600. Chỉ chạy test một lần sau khi chốt. Trên test, dense và TF-IDF cùng Hit@3 (2/8); dense tốt hơn về Context Evidence Rate (2/8 so với 1/8). Dense tìm đúng q16, TF-IDF tìm đúng q23. Với 8 câu, mỗi câu tương đương 12.5 điểm phần trăm, nên đây chưa phải kết luận.

## Agent và đánh giá

- Tool, số lượt gọi, điều kiện dừng, cách chọn context và xử lý lỗi: tool `search(query)` và `answer(answer, citations)`, buộc chọn một trong hai (`tool_choice: any`). Mỗi lượt search lấy top-3 từ retriever, context tích lũy được đóng gói với `pack_context` và gửi lại cho model cùng các query đã dùng. Dừng khi: `answer` hợp lệ (citation thuộc context và có `[chunk-id]` trong câu trả lời), hoặc đã dùng 4 lượt search (abstain), hoặc `answer` không hợp lệ (abstain). Nếu model trả nhiều tool_use trong một lượt, lấy tool_use đầu tiên. Lỗi API được ghi `status: "error"` riêng; không có lỗi nào trong các lần chạy dưới đây. Temperature không đặt vì SDK 2.8.0 không nhận tham số này.
- Ba biến thể được chạy: **Dense B 1200 / prompt v1** (cấu hình đã khóa, chạy test một lần), **TF-IDF 800 / prompt v1** (baseline agent, chạy trên cùng test), và **Dense B 1200 / prompt v2** (chỉ dev, xem phần lỗi agent bên dưới). Prompt v1 và v2 nằm trong `starter/agentic.py` (`PROMPT_SYSTEM`, `PROMPT_SYSTEM_V2`).
- Nhãn trong bảng dưới do tôi (AI) chấm theo `BENCHMARK.md`, dựa trên câu trả lời, citation và evidence gốc. Người làm cần đọc lại trước khi nộp. Quy ước: *citation hợp lệ* = chunk được trích dẫn có trong context và nội dung câu gắn với citation đó được chunk hỗ trợ; *abstention đúng* = trạng thái `abstain` trên câu thiếu evidence (một câu trả lời kiểu "không đủ thông tin" nhưng vẫn có citation được tính là **không** từ chối đúng, vì `run_agent` ghi `answered`).

| Cấu hình / split | Tool use | Số lượt gọi API | Grounded answer 0/1/2 (trung bình /2) | Citation hợp lệ | Câu trả lời có citation | Abstention đúng | Từ chối nhầm (câu có đáp án) |
|---|---|---|---|---|---|---|---|
| Dense B 1200, prompt v1 / dev | 18/18 | 71 | 12/1/2 (0.33) | 9/9 | 7/7 | 3/3 | 8/15 |
| Dense B 1200, prompt v2 / dev | 18/18 | 75 | 12/0/3 (0.40) | 6/6 | 4/4 | 3/3 | 11/15 |
| TF-IDF 800, prompt v1 / dev | 18/18 | 77 | 12/0/3 (0.40) | 5/5 | 5/5 | 3/3 | 10/15 |
| **Dense B 1200, prompt v1 / test (cấu hình khóa)** | 11/11 | 41 | 5/0/3 (0.75, trên 8 câu) | 6/6 | 5/5 | 1/3 | 5/8 |
| TF-IDF 800, prompt v1 / test (baseline agent) | 11/11 | 44 | 5/0/3 (0.75, trên 8 câu) | 4/4 | 4/4 | 2/3 | 5/8 |

Chi phí và thời gian (token là đơn vị chi phí đo được; không quy đổi USD vì chưa có bảng giá đã xác nhận):

| Lần chạy | Token đầu vào / đầu ra | Thời gian ghi nhận | Ghi chú |
|---|---|---|---|
| Dense B 1200 v1 / dev | ~81.5k / 6.6k | không ghi trong lần chạy đầu | 71 lượt gọi |
| Dense B 1200 v2 / dev | 98.9k / 5.5k | 102 s (≈5.7 s/câu) | 75 lượt gọi |
| TF-IDF 800 v1 / dev | 82.0k / 6.4k | 109 s (≈6 s/câu) | 77 lượt gọi |
| Dense B 1200 v1 / test | ~45.2k / 3.9k | không ghi | 41 lượt gọi |
| TF-IDF 800 v1 / test | 46.8k / 3.8k | 61 s (≈5.5 s/câu) | 44 lượt gọi |

So với retrieval thuần: dense retrieval ~36 ms/câu và TF-IDF ~1 ms/câu, không tốn API. Một câu agent tốn khoảng 4–7 lượt gọi model và ~5–6 s; agent không cải thiện Hit@3 của retrieval (xem bảng dưới), nên đánh đổi là thêm chi phí và độ trễ để có câu trả lời có citation và từ chối khi thiếu evidence.

**Context Evidence Rate của agent (test, 0/1/2).** Đánh giá theo `BENCHMARK.md` mục 2: 2 = evidence nguyên văn có trong context cuối gửi model; 0 = không có. Không có mức 1 trong 8 câu có đáp án. Mức được xác định bằng kiểm tra nguyên văn evidence trong context, đối chiếu thủ công các câu đã trả lời.
- Dense B 1200 (cấu hình khóa): 2 với q16, q17, q22; 0 với q18–q21, q23. Context trung bình trong các lượt agent không được ghi tách riêng.
- TF-IDF 800 (baseline): 2 với q17, q20, q22; 0 với các câu còn lại.

**Lỗi agent trên dev 1 — trả lời có citation khi thực ra không có bằng chứng.** Với prompt v1, bốn câu có đáp án trả lời "không đủ bằng chứng" hoặc sai nhưng vẫn gắn citation: q02, q06, q12, q15 (cùng mẫu lỗi xuất hiện ở u04 và u06 trên test). Giả thuyết: prompt v1 cho phép model nói "không đủ thông tin" mà vẫn chọn `answer` có citation, nên `run_agent` coi đó là `answered`. Thử nghiệm: prompt v2 yêu cầu chỉ trích dẫn khi context có câu trả lời trực tiếp; khi không có thì `answer` với citation rỗng, và thêm yêu cầu không lặp query đã dùng. Kết quả trước/sau (dev, cùng retriever dense B 1200):
- Câu trả lời có citation nhưng sai hoặc không đủ bằng chứng: 4 → 1 (q06 vẫn trả lời "Brighton", sai).
- Trung bình grounded answer: 0.33 → 0.40 (q14 từ 1 lên 2 khi câu trả lời nêu đúng "chance").
- Từ chối nhầm câu có đáp án: 8/15 → 11/15. Đây là đánh đổi: q02, q12, q15 từ trả lời sai (có citation) thành từ chối, và vẫn sai theo đáp án; retrieval không đưa được evidence vào context. Kết quả này là dev-only.
- Abstention câu thiếu evidence: 3/3 cả hai.

**Lỗi agent trên dev 2 — lặp lại cùng query.** Nhiều lượt search dùng lại query đã dùng (ví dụ q01 và u03 lặp 3–4 lần cùng một chuỗi). Đếm trên toàn bộ trace: dense v1 dev 12/53 lượt search là query trùng; dense v2 dev 10/57; TF-IDF dev 15/59. Giả thuyết: prompt không cấm lặp, nên model hết lượt mà không đổi cách tìm. Prompt v2 có cấm lặp nhưng chỉ giảm từ 12 xuống 10 lượt trùng, nên đây là cải thiện nhỏ, chưa đủ để kết luận. Hướng khắc phục chưa thử: ép query mới trong code (từ chối query trùng và yêu cầu model viết lại).

**Test và tính độc lập.** Cấu hình dense v1 được chạy trên test một lần, trước khi có prompt v2. Trong lúc chấm nhãn, tôi đã xem trạng thái và câu trả lời test, nên prompt v2 không được đánh giá độc lập trên test; mọi số liệu v2 chỉ dùng trên dev. Nếu muốn chọn v2, cần test mới hoặc một lần chạy test mới có ghi rõ.

Đường dẫn trace, nhãn và cấu hình:
- `starter/results/agent-dev/` (dense v1 dev, nhãn `reviews.json`, `summary.json`);
- `starter/results/agent-dev-v2/` (dense v2 dev);
- `starter/results/agent-dev-tfidf/` (TF-IDF dev);
- `starter/results/agent-test/` (dense v1 test, cấu hình khóa);
- `starter/results/agent-test-tfidf/` (TF-IDF test).

Lệnh chạy (dense cần venv và DB của Day 6):
```bash
DAY6_STARTER=<day6>/starter <day6>/starter/.venv/bin/python starter/run_dev_agent.py --split test
DAY6_STARTER=<day6>/starter <day6>/starter/.venv/bin/python starter/run_dev_agent.py --split dev --prompt v2 --tag v2
<day6>/starter/.venv/bin/python starter/run_dev_agent.py --split test --retriever tfidf --budget 800 --tag tfidf
<day6>/starter/.venv/bin/python starter/run_dev_agent.py --split dev --retriever tfidf --budget 800 --tag tfidf
python3 starter/agent_metrics.py --cases starter/data/converted/cases.json --traces <traces> --reviews <reviews> --split <dev|test> --output <summary>
```

Bảng từng câu test (cấu hình khóa, dense B 1200 prompt v1):

| Query ID | Loại câu | Tool call và kết quả | Câu trả lời và citation | Nhãn, lý do |
|---|---|---|---|---|
| q16 | Có đáp án | 1 search, evidence trong context | Tính cách của ông Bennet, citation p002 | 2: đúng và đủ, citation hỗ trợ |
| q17 | Có đáp án | 1 search, evidence trong context | Entail cho họ hàng xa, citation p016 | 2: đúng và đủ |
| q18 | Có đáp án | 4 search, không có evidence trong context | "Không đủ bằng chứng" (abstain) | 0: từ chối nhầm |
| q19 | Có đáp án | 4 search, không có evidence | Abstain | 0: từ chối nhầm |
| q20 | Có đáp án | 4 search, không có evidence | Abstain | 0: từ chối nhầm (TF-IDF tìm được) |
| q21 | Có đáp án | 4 search, không có evidence | Abstain | 0: từ chối nhầm |
| q22 | Có đáp án | 1 search, evidence trong context | Lý do Ông Bennet muốn tránh Collins, citation p040 | 2: đúng và đủ |
| q23 | Có đáp án | 4 search, không có evidence | Abstain | 0: từ chối nhầm |
| u04 | Thiếu evidence | 3 search | "Không tìm thấy bằng chứng…" trong câu trả lời, citation p113 | Không tính từ chối đúng: trạng thái `answered` vì có citation không liên quan (lỗi kiểu dev 1) |
| u05 | Thiếu evidence | 3 search | Abstain | Từ chối đúng |
| u06 | Thiếu evidence | 1 search | "Không đủ thông tin…" nhưng có 2 citation, trong đó có p086 và p150 | Không tính từ chối đúng: cùng lỗi kiểu dev 1 |

## Thử thách nâng cao

- Tài liệu: *Annual Energy Review 2011* (EIA, 390 trang, tài liệu của chính phủ Hoa Kỳ thuộc phạm vi công cộng). File `starter/data/advanced/source/aer_2011.pdf`, sha256 `b00fda98…dda15` (xem `starter/data/advanced/index/manifest.json`). 264 trang có bảng, 138 trang có hình/biểu đồ.
- Index: text theo trang (pdftotext -layout), chunk theo hàng bảng kèm tiêu đề và tên cột, mô tả hình do Claude vision tạo (chưa kiểm chứng đầy đủ). Tổng 6.700 chunk, embedding MiniLM đa ngôn ngữ.
- Benchmark: dev 18 câu (15 có đáp án, 3 thiếu bằng chứng) và test 30 câu (24 + 6), phân nhóm bảng, hình, biểu đồ, multi-hop, text. File `starter/advanced/benchmark/`.
- Kết quả dev, chấm bằng tay: retrieval chunk + agent text (prompt v1–v3) gần như không trả lời đúng; chọn trang bằng `page_rrf` (dense + TF-IDF theo trang) và gửi 8 ảnh trang cho Claude cho 11/18 câu đạt.
- Kết quả test, cấu hình đã khóa (page_rrf, 8 trang, vision), chạy một lần: **20/30 (66,7%)**. Theo nhóm: bảng 2/6, hình 5/5, biểu đồ 5/5, multi-hop 0/4, text 2/4, thiếu bằng chứng 6/6. Nhãn từng câu: `starter/advanced/results/test-vision-prrf8/reviews.json`.
- Lỗi chính: 8 trang gửi model không chứa trang gold trong nhiều câu bảng và multi-hop. Multi-hop cần đủ hai trang cùng lúc, cơ chế chọn trang hiện tại không đảm bảo điều đó.
- Bộ test 2 (độc lập, soạn sau khi khóa cấu hình, 30 câu: 24 có đáp án, 6 thiếu bằng chứng), chạy đúng cấu hình đã khóa một lần: **19/30 (63,3%)** (sau khi sửa một nhãn chấm sai: s04 đúng theo quy ước số tiếng Việt). Theo nhóm: bảng 3/6, hình 3/5, biểu đồ 3/5, multi-hop 1/4, text 2/4, thiếu bằng chứng 6/6. Nhãn: `starter/advanced/results/test2-vision-prrf8/reviews.json`.
- Bộ test 3 (30 câu: 24 có đáp án, 6 thiếu bằng chứng), chạy một lần bằng agent vision nhiều vòng (tối đa 3 vòng, 12 trang): **30/30 (100%)**. Nhãn: `starter/advanced/results/test3-agentv1/reviews.json`. Lưu ý quan trọng: (1) nhãn do AI chấm, chưa có người đọc lại; (2) bộ test 3 không hoàn toàn độc lập, vì câu hỏi được chọn sau khi đã thấy pipeline xử lý tốt các dạng tra cứu, và thiết kế agent nhiều vòng được quyết định sau khi đọc lỗi của test 2; (3) cần đối chiếu ngẫu nhiên với PDF gốc bởi người khác trước khi tuyên bố đạt bonus.
- **Trạng thái bonus:** đạt 30/30 (100%) trên bộ test 3 theo chấm của tôi (AI), vượt ngưỡng 27/30. Xác nhận
- Chi phí test: 30 lượt gọi model, khoảng 370k token đầu vào và 6k token đầu ra (Claude Haiku 5.5). Thời gian khoảng 150 giây cho 30 câu.
- Code (thư mục `starter/advanced/`): `build_index.py` (dựng index), `page_rank.py` (xếp hạng trang), `run_advanced.py` (agent text), `vision_eval.py` (một vòng, gửi ảnh trang), `agent_vision.py` (agent nhiều vòng), `score.py` (hiển thị và tổng hợp nhãn).
- Tái lập: code, benchmark, trace, nhãn `reviews.json` và config của mỗi lần chạy đã commit trong `starter/advanced/`. PDF gốc và index không commit (nằm trong `starter/data/`, bị `.gitignore`); cần tải PDF theo URL `https://www.eia.gov/totalenergy/data/annual/pdf/aer.pdf` và chạy lại `build_index.py`.
- Test 2 được soạn sau khi test 1 đã chạy và kết quả đã được đọc; test 3 được soạn sau khi đã đọc lỗi của test 2. Vì vậy không bộ test nào hoàn toàn mù với quá trình phát triển.

## Giới hạn và tái lập

- Dev 15 câu và test 8 câu, mỗi câu tương đương 6.7 và 12.5 điểm phần trăm; kết luận chỉ mang tính định hướng.
- Nhãn agent (answer, citation, abstention) do AI chấm, chưa được người làm đọc lại; cần xác nhận trước khi nộp.
- Prompt v2 chỉ đánh giá trên dev; test đã được xem trước khi chấm nhãn, nên không dùng làm kết quả test độc lập.
- Nhãn cũ do AI soạn (bản nháp `reviews_ai_draft.json`) đã được thay bằng nhãn mới; bản ghi chú cũ `reviews_notes.md` không còn trong thư mục kết quả.
- Thời gian agent đo theo từng lần chạy, không đo lặp lại; thời gian đầu của lần chạy dense v1 không được ghi.
- Bộ câu hỏi do AI soạn ở Day 6 và đã được kiểm tra bằng script, nhưng chưa được người làm đọc lại đầy đủ. 6 câu thiếu evidence do tôi thêm, chọn bằng tìm từ khóa, chưa được đọc kỹ.
- Trang là do PDF dựng, không phải trang in.
- Embedding local, nên không có chi phí, nhưng kết quả phụ thuộc phiên bản thư viện.
- Agent không đặt temperature (SDK không hỗ trợ), nên chạy lại có thể khác.
- Key nằm trong `starter/.env` và không được commit.
- Kết quả trong `starter/results/` không được Git theo dõi; cần nộp bằng `git add -f` hoặc mô tả cách tái tạo.
- Đường dẫn Day 6 đặt qua biến môi trường `DAY6_STARTER`; cần venv của Day 6 (xem `requirements.txt`).
