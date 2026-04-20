import streamlit as st
import requests
import pandas as pd
import time
import urllib.parse
import random
import json
import zipfile
import io
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- ML & Analysis Imports ---
from transformers import pipeline
from cleantext import clean
from bertopic import BERTopic
from bertopic.representation import MaximalMarginalRelevance
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

st.set_page_config(page_title="Universal Social Scraper & E2E Analyzer", layout="wide")

# ==========================================
# 🔒 SECURITY GATE (Hardcoded to "metdata")
# ==========================================
st.title("🔒 Pipeline Access")
pwd = st.text_input("Enter the pipeline password:", type="password")

if pwd != "metdata":
    if pwd: st.error("Incorrect password.")
    st.stop()

st.success("Access Granted! Welcome to the Metadata Pipeline.")
st.divider()

# ==========================================
# CONFIGURATION & SECRETS
# ==========================================
ANALYSIS_WORKERS = 15 

IG_API_KEY = st.secrets["RAPID_API_KEY_IG"]
IG_API_HOST = "instagram-scraper-stable-api.p.rapidapi.com"
IG_HEADERS = {"x-rapidapi-key": IG_API_KEY, "x-rapidapi-host": IG_API_HOST}

TK_API_KEY = st.secrets["RAPID_API_KEY_TK"]
TK_API_HOST = "tiktok381.p.rapidapi.com"
TK_HEADERS = {"x-rapidapi-key": TK_API_KEY, "x-rapidapi-host": TK_API_HOST}

genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
gemini_model = genai.GenerativeModel('gemini-1.5-flash')

@st.cache_resource
def load_hf_classifier():
    return pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None, truncation=True)

@st.cache_resource
def load_sentence_transformer():
    return SentenceTransformer("all-MiniLM-L6-v2")

classifier = load_hf_classifier()
embedding_model = load_sentence_transformer()

# ==========================================
# CORE FUNCTIONS
# ==========================================
def fetch_with_retry(url, headers, params=None, max_retries=5, log_container=None):
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, params=params, timeout=20)
            if response.status_code in [429, 403]:
                sleep_time = (2 ** attempt) + random.uniform(1, 3)
                if log_container: log_container.warning(f"⚠️ API Rate Limit. Pausing for {round(sleep_time, 1)}s...")
                time.sleep(sleep_time)
                continue
            response.raise_for_status()
            time.sleep(1) 
            return response.json()
        except Exception as e:
            time.sleep(2)
    return {}

def get_anger_score(text):
    if not text or not isinstance(text, str): return 0.0
    try:
        results = classifier(text[:1500])[0]
        return round(next(item['score'] for item in results if item['label'] == 'anger') * 100, 2)
    except: return 0.0

def process_rage_analysis(post_dict):
    """Calculates Rage for both IG and TK posts & comments"""
    text_key = "Post Text" if "Post Text" in post_dict else "Video Text"
    post_dict["Post Rage Score (%)"] = get_anger_score(post_dict.get(text_key, ""))
    
    raw_comments = post_dict.get("Raw Comments", [])
    avg_comment_rage = 0.0

    if raw_comments:
        c_rages = [get_anger_score(c["Comment Text"]) for c in raw_comments]
        for i, c in enumerate(raw_comments): c["Rage Score (%)"] = c_rages[i]
        avg_comment_rage = round(sum(c_rages) / len(c_rages), 2)

    post_dict["Comments Avg Rage Score (%)"] = avg_comment_rage
    return post_dict

def clean_social_text(text):
    if not isinstance(text, str): return ""
    return clean(text, fix_unicode=True, to_ascii=True, lower=True, no_line_breaks=True, 
                 no_urls=True, no_emails=True, no_phone_numbers=True, 
                 replace_with_url="<URL>", replace_with_email="<EMAIL>", replace_with_phone_number="<PHONE>")

def generate_deep_insights(raw_scraped_data, text_key):
    """Runs the BERTopic + Gemini Pipeline directly from raw data lists"""
    docs_to_process = []
    for post in raw_scraped_data:
        p_text = str(post.get(text_key, ""))
        for c in post.get("Raw Comments", []):
            c_text = str(c.get("Comment Text", ""))
            docs_to_process.append(f"POST CONTEXT: {p_text} | COMMENT: {c_text}")

    if not docs_to_process:
        return [{"Status": "No data found to analyze."}]

    cleaned_docs = [clean_social_text(doc) for doc in docs_to_process if doc]

    if len(cleaned_docs) < 15:
        return [{"Status": "Insufficient Data", "Message": f"Only found {len(cleaned_docs)} pieces of text. The AI requires at least 15 comments/posts to identify meaningful clusters."}]

    safe_topic_size = max(3, min(15, len(cleaned_docs) // 5)) 
    vectorizer_model = CountVectorizer(ngram_range=(1, 2), stop_words="english")
    representation_model = MaximalMarginalRelevance(diversity=0.5)

    try:
        topic_model = BERTopic(
            embedding_model=embedding_model, 
            min_topic_size=safe_topic_size,
            vectorizer_model=vectorizer_model,
            representation_model=representation_model
        )
        topics, probs = topic_model.fit_transform(cleaned_docs)
    except Exception as e:
        return [{"Status": "Clustering Error", "Message": "The AI could not form clusters. Data may lack variety or be too small.", "Error_Details": str(e)}]

    topic_info = topic_model.get_topic_info()
    valid_topics = topic_info[topic_info['Topic'] != -1].head(10)
    
    insights_report = []
    for index, row in valid_topics.iterrows():
        topic_id = row['Topic']
        size = row['Count']
        keywords = ", ".join([word[0] for word in topic_model.get_topic(topic_id)])
        docs_string = "\n".join([f"- {doc}" for doc in topic_model.get_representative_docs(topic_id)])

        prompt = f"""
        You are an expert marketing strategist and consumer intelligence analyst. Analyze the following social media data cluster.
        Cluster Keywords: {keywords}
        Representative Discussions: {docs_string}

        Extract the following information from this cluster and return it strictly as JSON format:
        {{
            "Audience_Segment": "Who is talking here? (2-5 words)",
            "Content_Hooks": "What specific phrases or slang should we use in ad copy to grab their attention? (1-2 sentences)",
            "Consumer_Pain_Point": "What specific problem are they trying to solve or complaining about? (1-2 sentences)",
            "Product_Opportunity": "How can our brand step in to fix this tension or market to them better? (1-2 sentences)"
        }}
        """
        try:
            response = gemini_model.generate_content(prompt)
            cleaned_response = response.text.replace('```json', '').replace('```', '').strip()
            insight_json = json.loads(cleaned_response)
            insight_json["Cluster_Size"] = size
            insight_json["Top_Keywords"] = keywords
            insights_report.append(insight_json)
        except: pass 
    
    return insights_report

def create_zip(files_dict):
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
        for file_name, file_data in files_dict.items():
            zip_file.writestr(file_name, file_data)
    return zip_buffer.getvalue()

# ==========================================
# PARALLEL PIPELINE FUNCTIONS
# ==========================================
def run_instagram_pipeline(ig_keywords, ig_post_limit, ig_comm_limit, master_log):
    master_log.write("🚀 **IG Pipeline Started!**")
    ig_raw_data = []
    generated_files = {}

    for kw in ig_keywords:
        master_log.write(f"🔍 **IG:** Searching '#{kw}'...")
        collected_posts = []
        pag_token = ""
        
        while len(collected_posts) < ig_post_limit:
            data = fetch_with_retry(f"https://{IG_API_HOST}/search_hashtag.php", IG_HEADERS, {"hashtag": kw, "pagination_token": pag_token}, log_container=master_log)
            batch = []
            if 'posts' in data and 'edges' in data['posts']: batch.extend(data['posts']['edges'])
            if len(collected_posts) == 0 and 'top_posts' in data and 'edges' in data['top_posts']: batch.extend(data['top_posts']['edges'])
            if not batch: break

            for edge in batch:
                node = edge.get('node', {})
                c_edges = node.get('edge_media_to_caption', {}).get('edges', [])
                
                # --- IG Post Date Extraction ---
                raw_ts = node.get('taken_at_timestamp')
                post_date = datetime.fromtimestamp(raw_ts).strftime('%Y-%m-%d %H:%M:%S') if raw_ts else "Unknown"

                collected_posts.append({
                    "Keyword": kw,
                    "Post URL": f"https://www.instagram.com/p/{node.get('shortcode')}/",
                    "shortcode": node.get('shortcode'),
                    "Post Date": post_date,
                    "Post Text": c_edges[0]['node']['text'] if c_edges else "",
                    "Total Comments": node.get('edge_media_to_comment', {}).get('count') or 0
                })
            pag_token = data.get('pagination_token')
            if not pag_token: break

        collected_posts = collected_posts[:ig_post_limit]
        master_log.write(f"📥 **IG:** Found {len(collected_posts)} posts for #{kw}. Extracting comments...")

        for post in collected_posts:
            extracted_comments = []
            if post['Total Comments'] > 0:
                c_pag = ""
                while len(extracted_comments) < ig_comm_limit:
                    c_data = fetch_with_retry(f"https://{IG_API_HOST}/get_post_comments.php", IG_HEADERS, {"media_code": post['shortcode'], "sort_order": "popular", "pagination_token": c_pag}, log_container=master_log)
                    r_comms = c_data.get('comments', [])
                    if not r_comms: 
                        if 'data' in c_data and isinstance(c_data['data'], list): r_comms = c_data['data']
                    if not r_comms: break

                    for c in r_comms:
                        if len(extracted_comments) >= ig_comm_limit: break
                        node = c.get('node', c)
                        
                        # --- IG Comment Date Extraction ---
                        c_ts = node.get('created_at')
                        comm_date = datetime.fromtimestamp(c_ts).strftime('%Y-%m-%d %H:%M:%S') if c_ts else "Unknown"

                        extracted_comments.append({
                            "Post URL": post['Post URL'],
                            "Comment ID": str(node.get('id', '')),
                            "Comment Date": comm_date,
                            "Username": f"@{node.get('user', {}).get('username', 'Unknown')}",
                            "Comment Text": str(node.get('text', '')).replace('\n', ' '),
                        })
                    c_pag = c_data.get('pagination_token') or c_data.get('next_max_id')
                    if not c_pag: break
                    time.sleep(1)

            post["Raw Comments"] = extracted_comments
            ig_raw_data.append(post)

    if ig_raw_data:
        ig_processed = []
        ig_all_comms = []
        master_log.write("🧠 **IG:** Running NLP & Extracting Insights...")
        
        with ThreadPoolExecutor(max_workers=ANALYSIS_WORKERS) as executor:
            for future in as_completed([executor.submit(process_rage_analysis, p) for p in ig_raw_data]):
                res = future.result()
                if res.get("Raw Comments"): ig_all_comms.extend(res["Raw Comments"])
                res_copy = res.copy()
                del res_copy["Raw Comments"]
                ig_processed.append(res_copy)

        ig_insights = generate_deep_insights(ig_raw_data, "Post Text")

        generated_files["IG_1_Posts.csv"] = pd.DataFrame(ig_processed).to_csv(index=False)
        if ig_all_comms: generated_files["IG_2_Comments.csv"] = pd.DataFrame(ig_all_comms).to_csv(index=False)
        if ig_insights: generated_files["IG_3_Marketing_Briefs.csv"] = pd.DataFrame(ig_insights).to_csv(index=False)

    master_log.write("🟢 **Instagram Pipeline Fully Complete!**")
    return generated_files


def run_tiktok_pipeline(tk_keywords, tk_region, tk_post_limit, tk_comm_limit, master_log):
    master_log.write("🚀 **TK Pipeline Started!**")
    tk_raw_data = []
    generated_files = {}

    for kw in tk_keywords:
        master_log.write(f"🔍 **TK:** Searching '{kw}'...")
        collected_vids = []
        cursor = 0

        while len(collected_vids) < tk_post_limit:
            data = fetch_with_retry(f"https://{TK_API_HOST}/search-video/", TK_HEADERS, {"keyword": kw, "count": "20", "cursor": str(cursor), "region": tk_region}, log_container=master_log)
            batch = data.get('videos') or data.get('aweme_list') or []
            if not batch: break
            
            collected_vids.extend(batch)
            cursor += 20
            if data.get('has_more') == 0: break
            time.sleep(1)

        collected_vids = collected_vids[:tk_post_limit]
        master_log.write(f"📥 **TK:** Found {len(collected_vids)} vids for '{kw}'. Extracting comments...")

        for vid in collected_vids:
            handle = vid.get('author', {}).get('unique_id', 'Unknown')
            v_url = f"https://www.tiktok.com/@{handle}/video/{vid.get('video_id') or vid.get('aweme_id')}"
            t_comments = vid.get('statistics', {}).get('comment_count', 0)
            
            # --- TK Post Date Extraction ---
            v_ts = vid.get('create_time')
            post_date = datetime.fromtimestamp(v_ts).strftime('%Y-%m-%d %H:%M:%S') if v_ts else "Unknown"
            
            extracted_comments = []
            if t_comments > 0:
                c_cursor = 0
                while len(extracted_comments) < tk_comm_limit:
                    c_data = fetch_with_retry(f"https://{TK_API_HOST}/post-comments/?url={urllib.parse.quote(v_url, safe='')}&cursor={c_cursor}", TK_HEADERS, log_container=master_log)
                    batch = c_data.get('comments', [])
                    if not batch: break

                    for c in batch:
                        if len(extracted_comments) >= tk_comm_limit: break
                        
                        # --- TK Comment Date Extraction ---
                        c_ts = c.get('create_time')
                        comm_date = datetime.fromtimestamp(c_ts).strftime('%Y-%m-%d %H:%M:%S') if c_ts else "Unknown"

                        extracted_comments.append({
                            "Video URL": v_url,
                            "Comment ID": c.get('id', ''),
                            "Comment Date": comm_date,
                            "Username": f"@{c.get('user', {}).get('unique_id', 'Unknown')}",
                            "Comment Text": str(c.get('text', '')).replace('\n', ' ')
                        })
                    c_cursor = c_data.get('cursor', 0)
                    if not c_data.get('hasMore', False): break
                    time.sleep(1)
            
            tk_raw_data.append({
                "Keyword": kw,
                "Video URL": v_url,
                "Post Date": post_date,
                "Video Text": vid.get('title') or vid.get('desc', ''),
                "Total Comments": t_comments,
                "Raw Comments": extracted_comments
            })

    if tk_raw_data:
        tk_processed = []
        tk_all_comms = []
        master_log.write("🧠 **TK:** Running NLP & Extracting Insights...")
        
        with ThreadPoolExecutor(max_workers=ANALYSIS_WORKERS) as executor:
            for future in as_completed([executor.submit(process_rage_analysis, p) for p in tk_raw_data]):
                res = future.result()
                if res.get("Raw Comments"): tk_all_comms.extend(res["Raw Comments"])
                res_copy = res.copy()
                del res_copy["Raw Comments"]
                tk_processed.append(res_copy)

        tk_insights = generate_deep_insights(tk_raw_data, "Video Text")

        generated_files["TK_1_Posts.csv"] = pd.DataFrame(tk_processed).to_csv(index=False)
        if tk_all_comms: generated_files["TK_2_Comments.csv"] = pd.DataFrame(tk_all_comms).to_csv(index=False)
        if tk_insights: generated_files["TK_3_Marketing_Briefs.csv"] = pd.DataFrame(tk_insights).to_csv(index=False)

    master_log.write("🟢 **TikTok Pipeline Fully Complete!**")
    return generated_files


# ==========================================
# APP UI & EXECUTION
# ==========================================
st.header("⚙️ 1. Pipeline Configuration")
st.markdown("Configure limits and upload your target keyword CSVs (first column will be read). Both platforms will run concurrently to save time.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📷 Instagram Scraper")
    ig_file = st.file_uploader("Upload IG CSV", type=['csv'], key="ig_up")
    ig_post_limit = st.number_input("Max Posts per Keyword (IG)", min_value=1, max_value=200, value=20)
    ig_comm_limit = st.number_input("Max Comments per Post (IG)", min_value=1, max_value=1000, value=50)

with col2:
    st.subheader("🎵 TikTok Scraper")
    tk_file = st.file_uploader("Upload TK CSV", type=['csv'], key="tk_up")
    tk_region = st.text_input("TikTok Region (e.g., US, GB)", value="US").strip().upper()
    tk_post_limit = st.number_input("Max Videos per Keyword (TK)", min_value=1, max_value=200, value=20)
    tk_comm_limit = st.number_input("Max Comments per Video (TK)", min_value=1, max_value=1000, value=50)

st.divider()

if st.button("🚀 Start End-to-End Pipeline", type="primary", use_container_width=True):
    
    if not ig_file and not tk_file:
        st.error("Please upload at least one CSV file to start.")
        st.stop()

    final_zip_files = {}
    master_log = st.expander("Live Pipeline Execution Logs", expanded=True)

    # Use ThreadPoolExecutor to run both platforms at the EXACT SAME TIME
    with st.spinner("Scraping and Analyzing in Parallel..."):
        with ThreadPoolExecutor(max_workers=2) as main_executor:
            futures = []
            
            if ig_file:
                ig_keywords = pd.read_csv(ig_file, header=None).iloc[:, 0].dropna().astype(str).tolist()
                futures.append(main_executor.submit(run_instagram_pipeline, ig_keywords, ig_post_limit, ig_comm_limit, master_log))
                
            if tk_file:
                tk_keywords = pd.read_csv(tk_file, header=None).iloc[:, 0].dropna().astype(str).tolist()
                futures.append(main_executor.submit(run_tiktok_pipeline, tk_keywords, tk_region, tk_post_limit, tk_comm_limit, master_log))
            
            # Wait for both to finish and collect all files
            for future in as_completed(futures):
                try:
                    result_files = future.result()
                    final_zip_files.update(result_files)
                except Exception as e:
                    master_log.error(f"❌ A critical error occurred in one of the pipelines: {e}")

    # ---------------------------------------------------------
    # FINAL EXPORT
    # ---------------------------------------------------------
    st.success("🎉 All Pipelines Complete! Data is ready for download.")
    
    if final_zip_files:
        st.download_button(
            label="📦 Download Complete Output (.zip)",
            data=create_zip(final_zip_files),
            file_name=f"E2E_Metadata_Export_{datetime.now().strftime('%Y%m%d_%H%M')}.zip",
            mime="application/zip",
            use_container_width=True
        )
