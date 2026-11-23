/**
 * Web Search Extension
 *
 * Adds a `websearch` tool usable by any model (including local ones).
 * Uses DuckDuckGo Lite (no API key) and returns parsed results.
 * Also adds a `webfetch` tool for reading a URL's text content.
 */

import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";

const SEARCH_PARAMS = Type.Object({
	query: Type.String({ description: "The search query" }),
});

const FETCH_PARAMS = Type.Object({
	url: Type.String({ description: "The URL to fetch" }),
});

const UA =
	"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0 Safari/537.36";

function htmlToText(html: string): string {
	return html
		.replace(/<script[\s\S]*?<\/script>/gi, " ")
		.replace(/<style[\s\S]*?<\/style>/gi, " ")
		.replace(/<[^>]+>/g, " ")
		.replace(/&amp;/g, "&")
		.replace(/&lt;/g, "<")
		.replace(/&gt;/g, ">")
		.replace(/&quot;/g, '"')
		.replace(/&#x27;|&#39;/g, "'")
		.replace(/&nbsp;/g, " ")
		.replace(/\s+/g, " ")
		.trim();
}

export default function webSearchExtension(pi: ExtensionAPI) {
	pi.registerTool({
		name: "websearch",
		label: "Web Search",
		description:
			"Search the web with DuckDuckGo. Returns the top results (title, URL, snippet). Use for current information, docs, CVE details, library APIs, anything you are unsure about.",
		promptSnippet: "Web search via DuckDuckGo for up-to-date information.",
		promptGuidelines: [
			"Use websearch when the user asks about recent events, specific versions, documentation, or anything you may not know.",
		],
		parameters: SEARCH_PARAMS,
		async execute(toolCallId, params, signal) {
			const q = encodeURIComponent(params.query);
			const res = await fetch(
				`https://lite.duckduckgo.com/lite/?q=${q}`,
				{ headers: { "User-Agent": UA }, signal },
			);
			const html = await res.text();

			// DDG lite: results are <a href=... class='result-link'> + <td class='result-snippet'>
			// (attribute order varies, quotes are single — match both)
			const results: { title: string; url: string; snippet: string }[] = [];
			const linkRe = /<a[^>]*href=["']([^"']+)["'][^>]*class=['"]result-link['"][^>]*>([\s\S]*?)<\/a>/g;
			const snippetRe = /<td[^>]*class=['"]result-snippet['"][^>]*>([\s\S]*?)<\/td>/g;
			const links = [...html.matchAll(linkRe)];
			const snippets = [...html.matchAll(snippetRe)].map((m) => htmlToText(m[1]));

			for (let i = 0; i < links.length && i < 8; i++) {
				let url = links[i][1];
				// DDG redirect wrapper
				const uddg = url.match(/[?&]uddg=([^&]+)/);
				if (uddg) url = decodeURIComponent(uddg[1]);
				results.push({
					title: htmlToText(links[i][2]),
					url,
					snippet: snippets[i] ?? "",
				});
			}

			const text = results.length
				? results
						.map((r, i) => `${i + 1}. ${r.title}\n   ${r.url}\n   ${r.snippet}`)
						.join("\n\n")
				: `No results for "${params.query}". Raw page may have changed.`;

			return {
				content: [{ type: "text", text }],
				details: { n: results.length },
			};
		},
	});

	pi.registerTool({
		name: "webfetch",
		label: "Web Fetch",
		description:
			"Fetch a URL and return its readable text content (HTML tags stripped). Use after websearch to read a promising result.",
		promptSnippet: "Fetch and read the text content of a web page.",
		promptGuidelines: [
			"Use webfetch to read pages found with websearch when the snippet is not enough.",
		],
		parameters: FETCH_PARAMS,
		async execute(toolCallId, params, signal) {
			const res = await fetch(params.url, {
				headers: { "User-Agent": UA },
				signal,
			});
			const html = await res.text();
			const text = htmlToText(html);
			return {
				content: [
					{
						type: "text",
						text: text.slice(0, 15000) || "(empty page)",
					},
				],
				details: { status: res.status, bytes: html.length },
			};
		},
	});
}
