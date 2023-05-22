# Use a Node.js base image from DockerHub
FROM node:14

# Set the working directory in the Docker container
WORKDIR /app

# Copy package.json and package-lock.json for npm install
COPY package*.json ./

# Install npm packages
RUN npm install

# Copy the rest of your application code
COPY . .

# Build your frontend
RUN npm run build

# Expose the port that your app runs on
EXPOSE 3000

# The command to run your application
CMD ["npm", "start"]
