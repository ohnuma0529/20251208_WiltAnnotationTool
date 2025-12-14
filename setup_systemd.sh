#!/bin/bash
set -e

echo "=== Installing Systemd Services ==="

# Backend
echo "Installing wilt-backend.service..."
sudo cp systemd/wilt-backend.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable wilt-backend
sudo systemctl restart wilt-backend

# Frontend
echo "Installing wilt-frontend.service..."
sudo cp systemd/wilt-frontend.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable wilt-frontend
sudo systemctl restart wilt-frontend

echo "=== Installation Complete ==="
echo "Services started:"
systemctl status wilt-backend wilt-frontend --no-pager | grep Active
