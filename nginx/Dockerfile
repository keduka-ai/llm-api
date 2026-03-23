FROM nginx:1.27.0-alpine3.19-slim

# Install curl using apk, the Alpine package manager
RUN apk add --no-cache curl

# Copy configuration files
COPY ./default.conf /etc/nginx/conf.d/default.conf
COPY ./nginx.conf /etc/nginx/nginx.conf


# COPY nginx.conf /etc/nginx/conf.d/default.conf
# Create a system group and user
# RUN addgroup -S zp && \
#     adduser -S -G zp -h /var/cache/nginx -s /sbin/nologin -D zp

# # Ensure necessary directories exist and have correct permissions
# RUN mkdir -p /var/cache/nginx /var/run /var/log/nginx && \
#     chown -R zp:zp /var/cache/nginx /var/run /var/log/nginx

# # Switch to the new user
# USER zp
