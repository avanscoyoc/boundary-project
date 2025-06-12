#!/bin/bash

# Initialize DB cluster if needed (only first run)
if [ ! -d "/var/lib/postgresql/15/main" ]; then
    echo "Initializing PostgreSQL database cluster..."
    pg_createcluster 15 main --start
fi

# Start PostgreSQL service
service postgresql start

# Create user and DB if not exist (run as postgres user)
sudo -u postgres psql -tc "SELECT 1 FROM pg_roles WHERE rolname='devuser'" | grep -q 1 || sudo -u postgres psql -c "CREATE USER devuser WITH PASSWORD 'devpassword';"
sudo -u postgres psql -tc "SELECT 1 FROM pg_database WHERE datname='devdb'" | grep -q 1 || sudo -u postgres psql -c "CREATE DATABASE devdb OWNER devuser;"
sudo -u postgres psql -d devdb -c "CREATE EXTENSION IF NOT EXISTS postgis;"

# Keep container running with bash
exec "$@"