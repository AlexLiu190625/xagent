# Xagent Docker Deployment

This directory contains Docker configuration files for deploying Xagent with Docker Compose.
Note: the base `docker-compose.yml` is located in the project root directory. Advanced
Compose overlays, including sandbox runtime options, live in this directory.

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Frontend      │     │    Backend      │     │   PostgreSQL   │
│  (Next.js)      │────│   (FastAPI)     │────│   Database      │
│  Port: 80       │ API │  Port: 8000      │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

### Services

- **Frontend**: Next.js standalone build served by nginx
- **Backend**: FastAPI with Python 3.11, Node.js 22, Playwright, LibreOffice
- **PostgreSQL**: PostgreSQL 16 database

## Quick Start

### 1. Configure Environment

Copy and edit the environment file:

```bash
cp example.env .env
# Edit .env with your API keys
```

Required environment variables:

```bash
# LLM API Keys (at least one required)
OPENAI_API_KEY="your-openai-api-key"
DEEPSEEK_API_KEY="your-deepseek-api-key"

# Database Password (auto-generated if using docker-compose)
POSTGRES_PASSWORD="xagent_password"
```

### 2. Start Services

From the project root directory:

```bash
docker compose up -d
```

This will start all services in the background.

### 3. Access Services

- **Frontend**: http://localhost:80
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

### 4. View Logs

```bash
# All services
docker compose logs -f

# Specific service
docker compose logs -f backend
docker compose logs -f frontend
docker compose logs -f postgres
```

### 5. Stop Services

```bash
docker compose down
```

## Advanced Usage

### Custom Port

By default, the frontend runs on port 80. To use a different port (e.g., 8080):

```bash
# In .env file
NGINX_PORT="8080"

# Then start
docker compose up -d
```

### Sandbox Runtime Overlays

Sandbox deployment is an advanced option. Use one sandbox overlay at a time, from
the project root.

Boxlite/KVM sandbox:

```bash
docker compose \
  -f docker-compose.yml \
  -f docker/docker-compose.sandbox.boxlite.yml \
  up -d
```

This requires Linux or WSL2 with KVM support and grants the backend container
KVM access.

Docker sibling sandbox:

```bash
docker compose \
  -f docker-compose.yml \
  -f docker/docker-compose.sandbox.docker.yml \
  up -d
```

This mounts the host Docker socket into the backend container so Xagent can
create sibling sandbox containers through the host Docker daemon. Treat this as
a privileged deployment mode: Docker socket access is effectively host-level
container control.

Docker sibling mode also resolves sandbox bind mounts on the Docker host. The
overlay defaults `XAGENT_SANDBOX_HOST_PROJECT_ROOT` to the current project root
and binds `${XAGENT_HOST_STORAGE_ROOT:-/root/.xagent}` to `/root/.xagent`. It
also passes `XAGENT_SANDBOX_HOST_STORAGE_ROOT` into the backend so sandbox
workspace mounts under `/root/.xagent` are translated back to the host storage
path before they reach the host Docker daemon. Override these values when the
host checkout or storage directory lives elsewhere:

```bash
XAGENT_SANDBOX_HOST_PROJECT_ROOT="$PWD" \
XAGENT_HOST_STORAGE_ROOT="$HOME/.xagent" \
docker compose \
  -f docker-compose.yml \
  -f docker/docker-compose.sandbox.docker.yml \
  up -d
```

In Docker sibling mode, `SANDBOX_VOLUMES` sources are host-side paths. Use
absolute host paths; relative paths and `~` are rejected instead of being
expanded inside the backend container.

## Docker Files

- `Dockerfile.backend` - Backend image (FastAPI, Python, Node.js)
- `Dockerfile.frontend` - Frontend image (Next.js, nginx)
- `../docker-compose.yml` - Base multi-service orchestration
- `docker-compose.sandbox.boxlite.yml` - Boxlite/KVM sandbox overlay
- `docker-compose.sandbox.docker.yml` - Docker sibling sandbox overlay
- `.dockerignore` - Backend build exclusions
- `.dockerignore.frontend` - Frontend build exclusions
- `nginx.conf` - Frontend nginx configuration
- `entrypoint.sh` - Backend startup script

## Building Individual Images

### Backend

Backend image dependencies are resolved from the committed `pyproject.toml` and
`uv.lock` during the Docker build. Keep `uv.lock` up to date before publishing;
the backend image build runs `uv sync --locked` for reproducible installs.

```bash
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -f docker/Dockerfile.backend \
  -t xprobe/xagent-backend:latest \
  --push .
```

### Frontend

```bash
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -f docker/Dockerfile.frontend \
  -t xprobe/xagent-frontend:latest \
  --push ./frontend
```

## Publishing Images

Images are published to Docker Hub under the `xprobe` organization:
- Backend: `xprobe/xagent-backend:latest`
- Frontend: `xprobe/xagent-frontend:latest`

Manual GitHub Container Registry publishing is also available through the GitHub Actions workflow:
- Backend: `ghcr.io/<owner>/xagent-backend`
- Frontend: `ghcr.io/<owner>/xagent-frontend`

### Publish to Docker Hub

From the `docker/` directory:

```bash
# Publish with default tag (latest)
PUSH=true ./publish.sh

# Publish with version tag
PUSH=true ./publish.sh v1.0.0

# Local single-platform build without pushing
PLATFORMS=linux/arm64 ./publish.sh
```

`publish.sh` behavior:

- `PUSH=true` (or `CI=true`) -> publish images (`--push`)
- local default (`PUSH=false`) -> local build only (`--load`, single platform)
- local multi-platform without push will fail fast with a hint

Or manually:

```bash
# Build and tag
docker buildx build --platform linux/amd64,linux/arm64 -f docker/Dockerfile.backend -t xprobe/xagent-backend:latest --push .
docker buildx build --platform linux/amd64,linux/arm64 -f docker/Dockerfile.frontend -t xprobe/xagent-frontend:latest --push ./frontend
```

> If Docker Buildx is not initialized locally, run:
>
> ```bash
> docker buildx create --use
> docker run --privileged --rm tonistiigi/binfmt --install all
> ```

### First Time Setup

1. **Create Docker Hub repositories** (one-time):
   - Go to https://hub.docker.com/
   - Create repositories: `xagent-backend` and `xagent-frontend`
   - Or they will be auto-created on first push

2. **Login to Docker Hub** (one-time):
   ```bash
   docker login
   ```

3. **Publish images** (on each release):
   ```bash
   ./docker/publish.sh
   ```

### Docker Hub Repositories

- https://hub.docker.com/r/xprobe/xagent-backend
- https://hub.docker.com/r/xprobe/xagent-frontend

### Automatic Publishing (GitHub Actions)

Images are automatically published to Docker Hub when you create a GitHub release.
You can also run the same workflow manually and optionally publish to GHCR.

**Setup (one-time):**

1. Configure GitHub secrets:
   - Go to repository Settings → Secrets and variables → Actions
   - Add `DOCKERHUB_USERNAME`: Your Docker Hub username
   - Add `DOCKERHUB_PASSWORD`: Your Docker Hub access token (not your password)
     - Create at: https://hub.docker.com/settings/security
     - Use "Read & Write" permissions for pushing images

2. Ensure Docker Hub repositories exist:
   - `xprobe/xagent-backend`
   - `xprobe/xagent-frontend`

**Publish on release:**

```bash
# Create a new release (triggers GitHub Actions)
git tag v1.0.0
git push origin v1.0.0
gh release create v1.0.0
```

GitHub Actions will:
- Build backend and frontend images
- Tag with version (e.g., `v1.0.0`, `v1.0`, `v1`, `latest`)
- Push to Docker Hub

### Manual GHCR Publish

1. Open the `Publish Docker Images` or `Nightly Build` workflow in GitHub Actions.
2. Run it manually with `push_to_ghcr=true`.
3. Leave `push_to_dockerhub=false` if you only want GHCR.

GitHub Actions will:
- Log in to GHCR with `GITHUB_TOKEN`
- Build backend and frontend images
- Push to `ghcr.io/<owner>/xagent-backend` and `ghcr.io/<owner>/xagent-frontend`

**Workflow file:** `.github/workflows/docker-publish.yml`

## Production Deployment

### Environment Variables

Key production variables:

```bash
# Database (set via docker-compose.yml)
DATABASE_URL="postgresql://xagent:password@postgres:5432/xagent"

# Security
ENCRYPTION_KEY="your-encryption-key"
```

### Volumes

Data persists in Docker volumes:

- `postgres_data` - PostgreSQL database
- `xagent_data` - User data (~/.xagent/)
- `xagent_uploads` - Uploaded files

### Backup

```bash
# Backup database
docker compose exec postgres pg_dump -U xagent xagent > backup.sql

# Restore database
docker compose exec -T postgres psql -U xagent xagent < backup.sql
```

## Troubleshooting

### Container Won't Start

```bash
# Check logs
docker compose logs backend

# Check health status
docker compose ps
```

### Database Connection Issues

```bash
# Verify postgres is running
docker compose exec postgres pg_isready -U xagent

# Check database logs
docker compose logs postgres
```

### Rebuild After Code Changes

```bash
# Rebuild specific service
docker compose build backend
docker compose up -d backend

# Rebuild all
docker compose build
docker compose up -d
```

## Development

### Running Tests in Docker

```bash
# Run backend tests
docker compose exec backend pytest

# Run with coverage
docker compose exec backend pytest --cov=src/xagent --cov-report=html
```

### Hot Reload (Development Mode)

For development with hot reload, use the standard setup instead of Docker:

```bash
# Backend (from project root)
python -m xagent.web.__main__

# Frontend (from frontend/)
cd frontend
npm run dev
```

## Security Notes

- Change default passwords in production
- Use `.env` file (never commit secrets)
- Enable SSL/TLS for production deployments
- Use Docker secrets for sensitive data
- Keep images updated with security patches
