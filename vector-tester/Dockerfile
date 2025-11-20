FROM node:20-alpine AS python-base
WORKDIR /app
RUN apk add --no-cache python3 py3-pip make g++ \
  && python3 -m venv /opt/venv \
  && /opt/venv/bin/pip install --no-cache-dir huggingface-hub pyyaml
ENV PATH="/opt/venv/bin:${PATH}"

FROM python-base AS deps
COPY vector-tester/package.json vector-tester/package-lock.json ./
RUN npm install

FROM python-base AS builder
COPY --from=deps /app/node_modules ./node_modules
COPY vector-tester ./
RUN npm run build

FROM python-base AS runner
WORKDIR /app
ENV NODE_ENV=production
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/.next ./.next
COPY --from=builder /app/package.json ./package.json
COPY --from=builder /app/next.config.mjs ./next.config.mjs
COPY --from=builder /app/public ./public
COPY --from=builder /app/data ./data
EXPOSE 3000
CMD ["npm", "run", "start"]

FROM python-base AS dev
