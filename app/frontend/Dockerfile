# Use a Node.js base image from DockerHub
FROM node:14 as build

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

# Start a new stage from the Node.js base image
FROM node:14 as production

# Set the working directory
WORKDIR /app

# Install serve
RUN npm install -g serve

# Copy from the 'build' stage the built app to the 'production' stage
COPY --from=build /app/dist .

# Expose the port that your app runs on
EXPOSE 5000

# The command to run your application
CMD ["serve", "-s", ".", "-l", "5000"]
