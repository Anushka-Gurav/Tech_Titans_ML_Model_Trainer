# Use Node 18 alpine (lightweight)
FROM node:18-alpine

# Set working directory
WORKDIR /app

# Copy package files first (for Docker cache)
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy rest of frontend code
COPY . .

# Build the React app
RUN npm run build

# Expose port
EXPOSE 3000

# Start preview server
CMD ["npm", "run", "preview"]