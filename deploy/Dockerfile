# FedSpeak portfolio page: one static HTML file served by nginx.
# Built on the shared box, which is linux/arm64.
FROM nginx:1.27-alpine

# Copy exactly the page. Never copy docs/ wholesale: it can hold notes that
# must not be served.
COPY docs/index.html /usr/share/nginx/html/index.html

RUN printf 'server {\n\
    listen 80;\n\
    listen [::]:80;\n\
    server_tokens off;\n\
    root /usr/share/nginx/html;\n\
    index index.html;\n\
    gzip on;\n\
    gzip_types text/html text/css application/javascript;\n\
    add_header X-Content-Type-Options nosniff always;\n\
    add_header Referrer-Policy strict-origin-when-cross-origin always;\n\
    location = / { add_header Cache-Control "no-cache" always; try_files /index.html =404; }\n\
    location / { try_files $uri =404; }\n\
}\n' > /etc/nginx/conf.d/default.conf

EXPOSE 80
HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
  CMD wget -qO- http://127.0.0.1/ > /dev/null || exit 1
