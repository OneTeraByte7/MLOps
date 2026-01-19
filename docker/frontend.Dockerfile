# Build stage
FROM node:18-alpine AS build
WORKDIR /app
COPY dashboard/react-frontend/package.json dashboard/react-frontend/package-lock.json ./
RUN npm ci --silent
COPY dashboard/react-frontend/ ./
RUN npm run build

# Production stage
FROM nginx:stable-alpine
COPY --from=build /app/dist /usr/share/nginx/html
# Optional: copy a custom nginx config if you want
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
