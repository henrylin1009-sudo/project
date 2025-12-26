import requests
import pandas as pd
import json
from sentence_transformers import SentenceTransformer, util
import streamlit as st

# --------------------------
# CONFIG
# --------------------------
NEWS_API_KEY = st.secrets["NEWS_API_KEY"]
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
TWITTER_API_KEY = st.secrets["TWITTER_API_KEY"]

GEMINI_MODEL = "models/gemini-2.5-flash"
NEWS_COUNT = 3
POLY_API_URL = "https://gamma-api.polymarket.com/events"

SOURCES = "bloomberg,financial-times,the-wall-street-journal,cnbc,business-insider,forbes,reuters,bbc-news,cnn"
KEYWORDS = "economy OR finance OR markets OR bitcoin OR crypto OR inflation OR politics"

MAX_PAGES = 5
SIMILARITY_THRESHOLD = 0.2  # below this, consider event irrelevant

model = SentenceTransformer('all-MiniLM-L6-v2')


# ======================================================
# 1️⃣ NEWS FETCH
# ======================================================
def get_top_international_news():
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": KEYWORDS,
        "sources": SOURCES,
        "pageSize": NEWS_COUNT,
        "language": "en",
        "sortBy": "publishedAt"
    }
    headers = {"X-Api-Key": NEWS_API_KEY}
    r = requests.get(url, params=params, headers=headers)
    if r.status_code != 200:
        print("Error fetching news:", r.status_code, r.text)
        return []

    articles = r.json().get("articles", [])
    news_list = []
    for a in articles:
        text = (a.get("title") or "") + " " + (a.get("description") or "")
        news_list.append(text)
    return news_list


# ======================================================
# 2️⃣ FETCH POLYMARKET EVENTS
# ======================================================
def fetch_events():
    params = {"closed": "false", "active": "true", "limit": 200}
    r = requests.get(POLY_API_URL, params=params)
    if r.status_code != 200:
        print("Error fetching Polymarket events:", r.status_code)
        return []

    events = r.json()
    cleaned = []

    for ev in events:
        active_markets = []
        for m in ev.get("markets", []):
            if m.get("active") and not m.get("closed"):
                p_raw = m.get("outcomePrices")
                o_raw = m.get("outcomes")
                prices = json.loads(p_raw) if isinstance(p_raw, str) else p_raw
                outcomes = json.loads(o_raw) if isinstance(o_raw, str) else o_raw

                formatted = []
                if outcomes and prices:
                    for name, price in zip(outcomes, prices):
                        formatted.append(f"{name}: ${float(price):.4f}")

                m["formatted_odds"] = " | ".join(formatted)
                active_markets.append(m)

        if active_markets:
            ev["markets"] = active_markets
            cleaned.append(ev)
    return cleaned


# ======================================================
# 3️⃣ SEMANTIC MATCH NEWS → EVENTS
# ======================================================
def top3_events_for_news(news_text, events):
    news_vec = model.encode(news_text, convert_to_tensor=True)
    titles = [e["title"] for e in events]
    vecs = model.encode(titles, convert_to_tensor=True)
    scores = util.cos_sim(news_vec, vecs)[0]
    for i, ev in enumerate(events):
        ev["similarity"] = float(scores[i])

    relevant_events = [e for e in events if e["similarity"] >= SIMILARITY_THRESHOLD]
    return sorted(relevant_events, key=lambda x: x["similarity"], reverse=True)[:3]


# ======================================================
# 4️⃣ GEMINI KEYWORD GENERATION PER MARKET
# ======================================================
def get_market_keywords(event_title, market_question):
    """Generate keywords using both event group title and market question."""
    text = f"{event_title} | {market_question}"
    url = f"https://generativelanguage.googleapis.com/v1/{GEMINI_MODEL}:generateContent"
    prompt = f"""
Extract 6–10 short, high-signal search keywords/phrases to find social media discussion about this market.
Return ONLY a JSON array like ["keyword1","keyword2"].
Text: {text}
"""
    data = {"contents": [{"parts": [{"text": prompt}]}]}
    params = {"key": GEMINI_API_KEY}
    r = requests.post(url, json=data, params=params)
    try:
        response_text = r.json()["candidates"][0]["content"]["parts"][0]["text"]
        keywords = json.loads(response_text)
        return [k.strip() for k in keywords if k.strip()]
    except:
        return []


# ======================================================
# 5️⃣ FETCH TWEETS PER MARKET
# ======================================================
def fetch_tweets_for_market(keywords, max_pages=5):
    all_tweets = []
    seen_texts = set()
    for kw in keywords:
        for query_type in ["Top", "Recent"]:
            cursor = ""
            for _ in range(max_pages):
                r = requests.get(
                    "https://api.twitterapi.io/twitter/community/get_tweets_from_all_community",
                    headers={"X-API-Key": TWITTER_API_KEY},
                    params={"query": kw, "queryType": query_type, "cursor": cursor}
                ).json()

                batch = r.get("tweets", [])
                if not batch:
                    break

                new_tweets = [t for t in batch if t['text'] not in seen_texts]
                all_tweets.extend(new_tweets)
                seen_texts.update([t['text'] for t in new_tweets])

                if not r.get("has_next_page"):
                    break
                cursor = r.get("next_cursor", "")
    return all_tweets


# ======================================================
# 6️⃣ GEMINI SENTIMENT ANALYSIS
# ======================================================
def analyze_with_gemini(question, tweets):
    text = "\n\n".join(
        [f"User: {t['author']['userName']}\nLikes:{t.get('likeCount',0)}\n{t['text']}" for t in tweets]
    )
    prompt = f"""
Market: "{question}"

Provide a concise sentiment read ONLY from the tweets:

1) Public Sentiment (positive / negative / mixed)
2) Does sentiment imply YES or NO?
3) Rough probability % from tweets only
Be short.
Tweets:
{text}
"""
    url = f"https://generativelanguage.googleapis.com/v1/{GEMINI_MODEL}:generateContent"
    r = requests.post(url,
                      json={"contents": [{"parts": [{"text": prompt}]}]},
                      params={"key": GEMINI_API_KEY})
    try:
        return r.json()["candidates"][0]["content"]["parts"][0]["text"]
    except:
        return "Gemini failed"


# ======================================================
# 7️⃣ GEMINI OVERALL INSIGHT PER NEWS
# ======================================================
def overall_news_insight(news_text, event_groups):
    summary_lines = []
    for ev in event_groups:
        total_volume = sum(float(m['volume']) for m in ev['markets'])
        end_date = ev['markets'][0]['endDate'] if ev['markets'] else "N/A"
        markets_info = "\n".join([f"{m['question']}\nOdds: {m['formatted_odds']}" for m in ev['markets']])
        sentiment = ev.get("gemini_sentiment", "No analysis")
        summary_lines.append(
            f"🎯 Event Group: {ev['title']}\nTotal Volume: ${total_volume:,.0f}\nEnd: {end_date}\n{markets_info}\nPublic Sentiment (Gemini):\n{sentiment}\n"
        )

    combined_text = f"News: {news_text}\n\n" + "\n".join(summary_lines)
    prompt = f"""
You are a market analyst. Based on the news and the following 3 event groups (with markets, market odds, and public sentiment analysis):

{combined_text}

Provide a **concise insight** connecting the market data and public sentiment to the news. Exclude unrelated events.
Be informative but short.
"""
    url = f"https://generativelanguage.googleapis.com/v1/{GEMINI_MODEL}:generateContent"
    r = requests.post(url,
                      json={"contents": [{"parts": [{"text": prompt}]}]},
                      params={"key": GEMINI_API_KEY})
    try:
        return r.json()["candidates"][0]["content"]["parts"][0]["text"]
    except:
        return "Gemini failed"


# ======================================================
# 8️⃣ STREAMLIT APP
# ======================================================
st.title("Polymarket News Sentiment Dashboard")
st.write("Fetch news → match Polymarket events → analyze public sentiment")

if st.button("Run Analysis"):
    with st.spinner("Fetching data..."):
        news_list = get_top_international_news()
        events = fetch_events()

    if not news_list:
        st.error("No news loaded. Check News API.")
    if not events:
        st.error("No Polymarket events loaded.")

    for i, news in enumerate(news_list, 1):
        st.markdown("---")
        st.subheader(f"📰 NEWS #{i}")
        st.write(news)

        top_events = top3_events_for_news(news, events)
        if not top_events:
            st.write("No strongly related Polymarket markets found.")
            continue

        # Run sentiment analysis for each event group
        for ev in top_events:
            with st.spinner(f"Analyzing: {ev['title']}"):
                keywords = get_market_keywords(ev["title"], ev["markets"][0]["question"])
                tweets = fetch_tweets_for_market(keywords)
                ev["gemini_sentiment"] = analyze_with_gemini(ev["title"], tweets)

            # Display event group info in simplified format
            total_volume = sum(float(m['volume']) for m in ev['markets'])
            end_date = ev['markets'][0]['endDate'] if ev['markets'] else "N/A"
            st.write(f"### 🎯 Event Group: {ev['title']}")
            st.write(f"- Similarity Score: `{ev['similarity']:.3f}`")
            st.write(f"- Total Volume: ${total_volume:,.0f}")
            st.write(f"- End: {end_date}")

            for m in ev["markets"]:
                st.write(f"**{m['question']}**")
                st.write(f"- Odds: {m['formatted_odds']}")

            st.write("**Public Sentiment (Gemini):**")
            st.write(ev["gemini_sentiment"])

        # Overall news insight
        with st.spinner("Creating combined market + sentiment insight..."):
            insight = overall_news_insight(news, top_events)

        st.success("Combined Insight:")
        st.write(insight)
