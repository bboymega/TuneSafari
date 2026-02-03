FROM node:22.21.0-alpine

WORKDIR /app

COPY /app/package*.json ./

RUN npm install

COPY app /app/

EXPOSE 3000

CMD npm run build && npm start
