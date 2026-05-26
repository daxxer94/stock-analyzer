# 📱 Mobile & Web Deployment Guide

This Streamlit app works in any browser, including on your phone.
Below are four deployment options from easiest to most flexible.

---

## ✅ Option 1 — Streamlit Community Cloud (Recommended, Free)

**Best for:** Permanent URL you can bookmark on your phone.

### Steps:
1. Create a free account at [github.com](https://github.com)
2. Create a new **public** repository, upload all files from this folder
3. Create a free account at [share.streamlit.io](https://share.streamlit.io)
4. Click **"New app"** → select your repo → set main file to `app.py`
5. Click **Deploy**

### Result:
- You get a permanent URL: `https://yourusername-stock-analyzer.streamlit.app`
- Open it on your phone — it works immediately
- **Add to home screen** on iOS (Safari → Share → Add to Home Screen) or Android (Chrome → Menu → Add to Home Screen)

### Notes:
- Free tier: app sleeps after 7 days of inactivity (wakes in ~30 seconds)
- GitHub repo must be public for free tier
- For private repos: upgrade to Streamlit Teams ($30/month)

---

## 🚂 Option 2 — Railway (Always-on, Free starter)

**Best for:** Always-on deployment without GitHub public repo requirement.

### Steps:
1. Create account at [railway.app](https://railway.app)
2. New Project → Deploy from GitHub repo (or upload directly)
3. Add environment variable / start command:
   ```
   streamlit run app.py --server.port $PORT --server.address 0.0.0.0
   ```
4. Railway auto-detects Python and installs `requirements.txt`
5. You get a permanent HTTPS URL

### Notes:
- Free: $5/month credit (usually enough for personal use)
- No sleep — always on
- Custom domain supported

---

## 🐳 Option 3 — Docker (Home Server / VPS / Synology NAS)

**Best for:** Full control, running on your own hardware.

### Run locally with Docker:
```bash
# Build the image
docker build -t stock-analyzer .

# Run it
docker run -d -p 8501:8501 --name stock-analyzer stock-analyzer

# Access at: http://localhost:8501
# Or from phone on same WiFi: http://your-computer-ip:8501
```

### Run on a VPS (e.g. DigitalOcean €5/mo, Hetzner €4/mo):
```bash
# On your VPS:
git clone https://github.com/yourusername/stock-analyzer.git
cd stock-analyzer
docker build -t stock-analyzer .
docker run -d -p 8501:8501 --restart=always stock-analyzer

# Access from phone: http://your-vps-ip:8501
```

### Run on Synology NAS:
1. Install Docker from Package Center
2. Upload the `Dockerfile` and app files via File Station
3. Build image via Docker UI → Run container → expose port 8501
4. Access from phone on home network: `http://nas-ip:8501`

---

## 📡 Option 4 — Local + ngrok (Quick phone testing)

**Best for:** Testing on your phone without any deployment.

```bash
# 1. Run the app locally
streamlit run app.py

# 2. In another terminal, install and run ngrok
# Download from: https://ngrok.com/download
ngrok http 8501

# 3. ngrok gives you a URL like: https://abc123.ngrok-free.app
# 4. Open that URL on your phone
```

### Notes:
- Free tier: URLs change each session
- Paid ngrok ($8/mo): permanent subdomain

---

## 📲 Making It Look Like a Native App

Once deployed anywhere with HTTPS:

### iPhone / iPad (Safari):
1. Open the app URL in Safari
2. Tap the **Share** button (box with arrow)
3. Scroll down → **"Add to Home Screen"**
4. Name it "Stock Analyzer" → **Add**
5. It now opens fullscreen like a native app

### Android (Chrome):
1. Open the app URL in Chrome
2. Tap the **three-dot menu** (⋮)
3. Tap **"Add to Home Screen"** or **"Install app"**
4. Confirm → app icon appears on home screen

---

## 🔒 Security Notes

- The app has **no authentication** by default
- For private use, add Streamlit's built-in auth or deploy behind a VPN
- **Never expose port 8501 publicly without HTTPS** (Options 1–3 handle this automatically)
- For home use on your local network only, Docker without HTTPS is fine

---

## ⚡ Performance Tips

| Deployment | Cold start | Data latency | Cost |
|---|---|---|---|
| Streamlit Cloud | ~30s (if sleeping) | Normal | Free |
| Railway | None | Normal | ~Free |
| Docker/VPS | None | Fastest | €4–5/mo |
| ngrok (local) | None | Fastest | Free |

All data is **cached for 1 hour** by `@st.cache_data`, so repeated analysis of the same tickers is fast.
