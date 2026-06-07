To Run the RAG Files , create a virtual env.
cmd : python -m venv venv
Activate : source venv/bin/activate
Install from requirements.txt 
Cmd: pip install -r requirements.txt
Install redis container locally using docker.
Cmd: docker run -d -p 6379:6379 redis
Check if its running using "docker ps"
create your own GROQ_API_KEY and set it in .env file.
To Run the files , use the terminal and execute the files .
Some example cmds are as follows:
python3 RAG_Redis.py --question "I need a Speaker for beach party" --category "Speakers"
python3 RAG_HYBRID_SEARCH.py --question "I need a Speaker for beach party ” --category "Speakers"
python3 RAG_BM25.py --question "I need a Speaker for beach party ” --category "Speakers"
python3 RAG.py --question "I need a Speaker for beach party ” --category "Speakers" 
You can pass custom query using --question and category using --category switch.
If you wish to change the llm , you can do so by just changing the model (model="llama-3.3-70b-versatile") , in each file .



Example Output for RAG_Redis.py

bishakhasingh@Bishakhas-MacBook-Pro Intelligent-E-Commerce-Search-with-Retrieval-Augmented-Generation % python RAG_Redis.py --question "I need a Speaker for beach party" --category "Speakers"
/Users/bishakhasingh/Documents/Intelligent-E-Commerce-Search-with-Retrieval-Augmented-Generation/RAG_Redis.py:10: DeprecationWarning: Importing from redisvl.extensions.llmcache is deprecated. Please import from redisvl.extensions.cache.llm instead.
  from redisvl.extensions.llmcache import SemanticCache
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Loading weights: 100%|███████████████████████████████████████████████████████████████████████████████████| 103/103 [00:00<00:00, 14732.92it/s]
[Redis] Connected ✅ — redis://localhost:6379
Redis semantic cache enabled.
Loading weights: 100%|████████████████████████████████████████████████████████████████████████████████████| 393/393 [00:00<00:00, 6753.75it/s]
[Redis] Context store initialized ✅
[Redis] Client connected ✅
[Cache] redis_store=✅
[Cache] redis_client=✅
[Cache] Index has 4 docs
[Cache] Top similarity: 1.000 (threshold: 0.92)
[Cache HIT] key=ctx:0189d5da8211fe46c263ccd0d05ce1b1
/Users/bishakhasingh/Documents/Intelligent-E-Commerce-Search-with-Retrieval-Augmented-Generation/.venv/lib/python3.12/site-packages/langchain_redis/cache.py:686: LangChainPendingDeprecationWarning: The default value of `allowed_objects` will change in a future version. Pass an explicit list of allowed classes (or 'messages' for untrusted input that contains only chat messages) to suppress this warning.
  loads(gen_str)

=== ANSWER ===
**Beach Party Speaker Recommendations**

We've got just the right speakers for your beach party. Our top picks are designed to be portable, waterproof, and packed with features to keep the party going. Here are our recommendations:

1. **Landmark LM TBS7021 20W Wireless Bluetooth Party Speaker**:
        * Price: $87.99
        * Rating: 5.0 out of 5
        * Key Features: 20W wireless Bluetooth speaker, mic, handsfree calling, splashproof, 4000mAh battery
        * Benefits: Perfect for a beach party, this speaker is waterproof, compact, and has a long-lasting battery life. The mic and handsfree calling features make it easy to take calls or sing along to your favorite tunes.
2. **JBL PartyBox Encore Essential**:
        * Price: $329.99
        * Rating: 4.5 out of 5
        * Key Features: 100W monstrous pro sound, dynamic light show, up to 6 hours of playtime
        * Benefits: If you're looking for a more powerful speaker to get the party started, the JBL PartyBox Encore Essential is an excellent choice. The dynamic light show adds an extra layer of fun to your beach party, and the 100W sound ensures everyone can hear the music.

While the **Zepad Wireless Bluetooth Speaker** is also a portable option, its lower rating and lower sound quality make it less suitable for a beach party.

**Ultimate Recommendation**: If budget is not a concern, the **JBL PartyBox Encore Essential** is the way to go for its exceptional sound quality and dynamic light show. However, if you're looking for a more affordable option, the **Landmark LM TBS7021 20W Wireless Bluetooth Party Speaker** is a great value for its price, offering a perfect balance of sound quality, portability, and waterproofing.

=== PRODUCTS ===
No products found (check product_db and category filter).

=== CONTEXTS ===
product: landmark lm tbs7021 20w wireless bluetooth party speaker with mic & handsfree calling/ splashproof/ 4000mah battery - (bla.... category: speakers. price: $87.99. rating: 5.0 out of 5. reviews: 3. landmark lm tbs7021 20w wireless bluetooth party speaker with mic & handsfree calling/ splashproof/ 4000mah battery - (bla...
product: jbl partybox encore essential | portable bluetooth party speaker | 100w monstrous pro sound | dynamic light show | upto 6h.... category: speakers. price: $329.99. rating: 4.5 out of 5. reviews: 136. jbl partybox encore essential | portable bluetooth party speaker | 100w monstrous pro sound | dynamic light show | upto 6h...
product: zepad wireless bluetooth speaker portable speaker with mic super bass splashproof for house party dance (army color). category: speakers. price: $32.99. rating: 2.5 out of 5. reviews: 22. zepad wireless bluetooth speaker portable speaker with mic super bass splashproof for house party dance (army color)


## Evaluation: Faithfulness and Latency

Run the RAG pipelines first
python run_all_pipelines.py

This creates files such as:
rag_outputs_RAG_BM25.json
rag_outputs_RAG.json
rag_outputs_RAG_HYBRID_SEARCH.json
rag_outputs_RAG_Redis.json

1. Calculate Faithfulness
python score_faithfulness.py RAG_BM25
python score_faithfulness.py RAG
python score_faithfulness.py RAG_HYBRID_SEARCH
python score_faithfulness.py RAG_Redis

This generates:

faithfulness_RAG_BM25.json
faithfulness_RAG.json
faithfulness_RAG_HYBRID_SEARCH.json
faithfulness_RAG_Redis.json

and CSV files for plotting:

faithfulness_scores_RAG_BM25.csv
faithfulness_scores_RAG.csv
faithfulness_scores_RAG_HYBRID_SEARCH.csv
faithfulness_scores_RAG_Redis.csv

2. Calculate Latency

Run:
python latency_test.py

This uses:
test_queries10.json

and generates:
latency_results.json

3. Generate Plots
Run:
python plot_results.py

This generates figures such as:
answer_relevancy_bar.png

Correct Order:
python run_all_pipelines.py
python score_faithfulness.py RAG_BM25
python score_faithfulness.py RAG
python score_faithfulness.py RAG_HYBRID_SEARCH
python score_faithfulness.py RAG_Redis
python latency_test.py
python plot_results.py


