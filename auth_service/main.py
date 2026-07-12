import os
import secrets

import bcrypt
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import db

FRONTEND_ORIGIN = os.environ["FRONTEND_ORIGIN"]
SESSION_TTL_DAYS = int(os.environ.get("SESSION_TTL_DAYS", "7"))
COOKIE_SECURE = os.environ.get("COOKIE_SECURE", "false").lower() == "true"
COOKIE_SAMESITE = os.environ.get("COOKIE_SAMESITE", "lax")
COOKIE_NAME = "session_token"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_ORIGIN],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)


class Credentials(BaseModel):
    username: str = Field(min_length=3, max_length=32)
    password: str = Field(min_length=8, max_length=128)


@app.on_event("startup")
def on_startup() -> None:
    db.init_db()


def _create_session(user_id: int) -> str:
    token = secrets.token_urlsafe(32)
    with db.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO sessions (token, user_id, expires_at)
                VALUES (%s, %s, now() + %s * interval '1 day')
                """,
                (token, user_id, SESSION_TTL_DAYS),
            )
    return token


def _set_session_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        key=COOKIE_NAME,
        value=token,
        httponly=True,
        samesite=COOKIE_SAMESITE,
        secure=COOKIE_SECURE,
        path="/",
        max_age=SESSION_TTL_DAYS * 86400,
    )


def _current_username(request: Request) -> str | None:
    token = request.cookies.get(COOKIE_NAME)
    if not token:
        return None
    with db.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT users.username
                FROM sessions
                JOIN users ON users.id = sessions.user_id
                WHERE sessions.token = %s AND sessions.expires_at > now()
                """,
                (token,),
            )
            row = cur.fetchone()
    return row[0] if row else None


@app.post("/register", status_code=201)
def register(credentials: Credentials, response: Response):
    password_hash = bcrypt.hashpw(credentials.password.encode(), bcrypt.gensalt()).decode()
    with db.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO users (username, password_hash)
                VALUES (%s, %s)
                ON CONFLICT (username) DO NOTHING
                RETURNING id
                """,
                (credentials.username, password_hash),
            )
            row = cur.fetchone()
    if row is None:
        raise HTTPException(status_code=409, detail="Username is already taken")

    token = _create_session(row[0])
    _set_session_cookie(response, token)
    return {"username": credentials.username}


@app.post("/login")
def login(credentials: Credentials, response: Response):
    with db.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, password_hash FROM users WHERE username = %s",
                (credentials.username,),
            )
            row = cur.fetchone()

    if row is None or not bcrypt.checkpw(credentials.password.encode(), row[1].encode()):
        raise HTTPException(status_code=401, detail="Invalid username or password")

    token = _create_session(row[0])
    _set_session_cookie(response, token)
    return {"username": credentials.username}


@app.post("/logout")
def logout(request: Request, response: Response):
    token = request.cookies.get(COOKIE_NAME)
    if token:
        with db.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM sessions WHERE token = %s", (token,))
    response.delete_cookie(key=COOKIE_NAME, path="/", samesite=COOKIE_SAMESITE)
    return {"ok": True}


@app.get("/me")
def me(request: Request):
    username = _current_username(request)
    if username is None:
        raise HTTPException(status_code=401, detail="Not signed in")
    return {"username": username}
