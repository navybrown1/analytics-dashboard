# Deployment Guide

How to deploy the Business Analytics Dashboard to various platforms.

## Local Development

```bash
# Install dependencies
pip install -r requirements_enhanced.txt

# Run locally (uses .streamlit/config.toml: 127.0.0.1:8502)
streamlit run app_enhanced.py

# Or be explicit about port and address
streamlit run app_enhanced.py --server.address 127.0.0.1 --server.port 8502

# Run with custom settings (e.g. expose to LAN)
streamlit run app_enhanced.py \
  --server.port 8502 \
  --server.address 0.0.0.0 \
  --server.maxUploadSize 200
```

## Streamlit Community Cloud (Free)

The easiest way to deploy for free.

### Steps:

1. **Push to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial commit: Business Analytics Dashboard"
   git remote add origin https://github.com/YOUR_USERNAME/analytics-dashboard.git
   git push -u origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click "New app"
   - Select your repository
   - Set main file to `app_enhanced.py`
   - Click "Deploy"

3. **Configuration** (optional)
   Create `.streamlit/config.toml`:
   ```toml
   [server]
   maxUploadSize = 200
   
   [theme]
   primaryColor = "#1f77b4"
   backgroundColor = "#ffffff"
   secondaryBackgroundColor = "#f8f9fa"
   textColor = "#2c3e50"
   ```

## Docker Deployment

### Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements_enhanced.txt .
RUN pip install --no-cache-dir -r requirements_enhanced.txt

COPY . .

EXPOSE 8502

HEALTHCHECK CMD curl --fail http://localhost:8502/_stcore/health

ENTRYPOINT ["streamlit", "run", "app_enhanced.py", \
            "--server.port=8502", \
            "--server.address=0.0.0.0"]
```

### Build and Run

```bash
# Build
docker build -t analytics-dashboard .

# Run
docker run -p 8502:8502 analytics-dashboard

# Run in background
docker run -d -p 8502:8502 --name dashboard analytics-dashboard
```

### Docker Compose

```yaml
version: '3.8'
services:
  dashboard:
    build: .
    ports:
      - "8502:8502"
    environment:
      - STREAMLIT_SERVER_MAX_UPLOAD_SIZE=200
    restart: unless-stopped
```

```bash
docker compose up -d
```

## Heroku Deployment

### Setup Files

**Procfile:**
```
web: streamlit run app_enhanced.py --server.port $PORT --server.address 0.0.0.0
```

**runtime.txt:**
```
python-3.11.7
```

### Deploy

```bash
heroku create analytics-dashboard
git push heroku main
heroku open
```

## AWS EC2 Deployment

```bash
# SSH into your EC2 instance
ssh -i your-key.pem ec2-user@your-ip

# Install Python and dependencies
sudo yum update -y
sudo yum install python3-pip -y
pip3 install -r requirements_enhanced.txt

# Run with nohup (persists after disconnect)
nohup streamlit run app_enhanced.py \
  --server.port 8502 \
  --server.address 0.0.0.0 &

# Or use systemd for production
sudo tee /etc/systemd/system/dashboard.service << EOF
[Unit]
Description=Analytics Dashboard
After=network.target

[Service]
User=ec2-user
WorkingDirectory=/home/ec2-user/analytics-dashboard
ExecStart=/usr/local/bin/streamlit run app_enhanced.py --server.port 8502 --server.address 0.0.0.0
Restart=always

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl enable dashboard
sudo systemctl start dashboard
```

## Google Cloud Run

```bash
# Build and push container
gcloud builds submit --tag gcr.io/PROJECT_ID/analytics-dashboard

# Deploy
gcloud run deploy analytics-dashboard \
  --image gcr.io/PROJECT_ID/analytics-dashboard \
  --platform managed \
  --allow-unauthenticated \
  --port 8502
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `STREAMLIT_SERVER_PORT` | Server port | 8502 |
| `STREAMLIT_SERVER_ADDRESS` | Bind address | 127.0.0.1 |
| `STREAMLIT_SERVER_MAX_UPLOAD_SIZE` | Max upload MB | 200 |
| `STREAMLIT_BROWSER_GATHER_USAGE_STATS` | Usage stats | true |

## Security Considerations

- Always use HTTPS in production (use a reverse proxy like nginx)
- Set `STREAMLIT_SERVER_ENABLE_CORS=false` for production
- Consider adding authentication for sensitive data
- Limit upload file size based on your server capacity
- Monitor memory usage with large datasets

## Performance Tips

- Enable caching (already built into `app_enhanced.py`)
- Use a reverse proxy (nginx) for production traffic
- Set appropriate upload size limits
- Consider a CDN for static assets
- Monitor with tools like Prometheus/Grafana
