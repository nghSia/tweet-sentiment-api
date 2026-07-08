"""Reusable MySQL helpers for the tweets table."""

from __future__ import annotations

from typing import Iterable, Mapping


def fetch_all_tweets(connection):
	cursor = connection.cursor(dictionary=True)
	try:
		cursor.execute(
			"SELECT id, text, positive, negative FROM tweets ORDER BY id"
		)
		return cursor.fetchall()
	finally:
		cursor.close()


def count_tweets(connection):
	cursor = connection.cursor()
	try:
		cursor.execute("SELECT COUNT(*) FROM tweets")
		row = cursor.fetchone()
		return row[0] if row else 0
	finally:
		cursor.close()


def clear_tweets(connection):
	cursor = connection.cursor()
	try:
		cursor.execute("TRUNCATE TABLE tweets")
		connection.commit()
	except Exception:
		connection.rollback()
		raise
	finally:
		cursor.close()


def insert_tweet(connection, text, positive, negative):
	cursor = connection.cursor()
	try:
		cursor.execute(
			"INSERT INTO tweets (text, positive, negative) VALUES (%s, %s, %s)",
			(text, int(positive), int(negative)),
		)
		connection.commit()
		return cursor.lastrowid
	except Exception:
		connection.rollback()
		raise
	finally:
		cursor.close()


def insert_many_tweets(connection, tweets):
	cursor = connection.cursor()
	rows = [
		(tweet["text"], int(tweet["positive"]), int(tweet["negative"]))
		for tweet in tweets
	]
	try:
		cursor.executemany(
			"INSERT INTO tweets (text, positive, negative) VALUES (%s, %s, %s)",
			rows,
		)
		connection.commit()
		return cursor.rowcount
	except Exception:
		connection.rollback()
		raise
	finally:
		cursor.close()


def insert_many_from_dicts(connection, tweets: Iterable[Mapping[str, object]]):
	cursor = connection.cursor()
	rows = [
		(str(tweet["text"]), int(tweet["positive"]), int(tweet["negative"]))
		for tweet in tweets
	]
	try:
		cursor.executemany(
			"INSERT INTO tweets (text, positive, negative) VALUES (%s, %s, %s)",
			rows,
		)
		connection.commit()
		return cursor.rowcount
	except Exception:
		connection.rollback()
		raise
	finally:
		cursor.close()
