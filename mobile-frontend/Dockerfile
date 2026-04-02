# ─── Stage 1: Dependencies ───────────────────────────────────────────────────
FROM node:20-alpine AS deps

WORKDIR /app

# Install system dependencies required for native modules
RUN apk add --no-cache \
    python3 \
    make \
    g++ \
    git

COPY package.json .

RUN npm install --legacy-peer-deps

# ─── Stage 2: Development ────────────────────────────────────────────────────
FROM node:20-alpine AS development

WORKDIR /app

RUN apk add --no-cache \
    python3 \
    make \
    g++ \
    git \
    curl

# Copy installed node_modules from deps stage
COPY --from=deps /app/node_modules ./node_modules

# Copy all project files
COPY . .

# Expose Expo dev server ports
# 8081 = Metro bundler
# 19000 = Expo Go tunnel
# 19001 = Expo DevTools
EXPOSE 8081 19000 19001

ENV NODE_ENV=development
ENV EXPO_DEVTOOLS_LISTEN_ADDRESS=0.0.0.0
ENV REACT_NATIVE_PACKAGER_HOSTNAME=0.0.0.0

CMD ["npx", "expo", "start", "--web", "--host", "0.0.0.0"]

# ─── Stage 3: Web Production Build ───────────────────────────────────────────
FROM node:20-alpine AS builder

WORKDIR /app

RUN apk add --no-cache \
    python3 \
    make \
    g++ \
    git

COPY --from=deps /app/node_modules ./node_modules
COPY . .

ARG API_BASE_URL=http://localhost:5000
ENV EXPO_PUBLIC_API_BASE_URL=$API_BASE_URL
ENV NODE_ENV=production

RUN npx expo export --platform web --output-dir /app/dist

# ─── Stage 4: Production Web Server ──────────────────────────────────────────
FROM nginx:1.25-alpine AS production

COPY --from=builder /app/dist /usr/share/nginx/html

COPY docker/nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD wget -qO- http://localhost:80/ || exit 1

CMD ["nginx", "-g", "daemon off;"]
