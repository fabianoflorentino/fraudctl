#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

get_docker_compose() {
    if docker compose version >/dev/null 2>&1; then
        echo "docker compose"
    elif command -v docker-compose >/dev/null 2>&1; then
        echo "docker-compose"
    else
        echo "Error: Neither 'docker compose' nor 'docker-compose' is available."
        echo "Install docker-compose-plugin or docker-compose to use this command."
        exit 1
    fi
}

DOCKER_COMPOSE="$(get_docker_compose)"
cd "$PROJECT_DIR"

case "${1:-up}" in
    up)
        $DOCKER_COMPOSE up -d
        ;;
    down)
        $DOCKER_COMPOSE down
        ;;
    logs)
        $DOCKER_COMPOSE logs -f
        ;;
    build)
        $DOCKER_COMPOSE build
        ;;
    ps)
        $DOCKER_COMPOSE ps
        ;;
    restart)
        $DOCKER_COMPOSE down && $DOCKER_COMPOSE up -d
        ;;
    *)
        echo "Usage: $0 {up|down|logs|build|ps|restart}"
        exit 1
        ;;
esac
