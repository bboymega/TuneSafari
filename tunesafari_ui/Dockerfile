FROM node:22.21.0-alpine

WORKDIR /app

ARG DEBIAN_FRONTEND=noninteractive

COPY package*.json ./

RUN npm install

COPY . /app/

EXPOSE 3000

CMD ["sh", "-c", "npm run build && npm start"]
