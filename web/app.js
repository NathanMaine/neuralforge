/* NeuralForge Dashboard Application */

(function () {
    "use strict";

    const API = "";

    // --- Fetch helpers ---
    async function fetchJSON(url) {
        try {
            const resp = await fetch(API + url);
            if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
            return await resp.json();
        } catch (err) {
            console.error("Fetch error:", url, err);
            return null;
        }
    }

    // --- Dashboard ---
    async function loadDashboard() {
        const data = await fetchJSON("/api/dashboard");
        if (!data) return;

        document.getElementById("stat-nodes").textContent = data.graph.total_nodes;
        document.getElementById("stat-edges").textContent = data.graph.total_edges;
        document.getElementById("stat-chunks").textContent = data.knowledge_base.total_chunks;

        // Node types
        const typesContainer = document.getElementById("node-types");
        typesContainer.innerHTML = "";
        const counts = data.graph.node_type_counts || {};
        for (const [type, count] of Object.entries(counts)) {
            const badge = document.createElement("div");
            badge.className = "type-badge";
            badge.innerHTML = `${type}<span class="type-count">${count}</span>`;
            typesContainer.appendChild(badge);
        }

        // Status indicator
        const dot = document.getElementById("status-dot");
        const qs = data.knowledge_base.qdrant_status;
        dot.className = "status-indicator " + (qs === "green" ? "healthy" : qs === "yellow" ? "degraded" : "down");
    }

    // --- Experts ---
    async function loadExperts() {
        const data = await fetchJSON("/api/experts");
        if (!data) return;

        const list = document.getElementById("experts-list");
        list.innerHTML = "";

        document.getElementById("stat-experts").textContent = data.experts.length;

        if (data.experts.length === 0) {
            list.innerHTML = '<p class="no-data">No experts found. Ingest some data to get started.</p>';
            return;
        }

        for (const expert of data.experts) {
            const card = document.createElement("div");
            card.className = "expert-card";
            card.innerHTML = `
                <h3>${escapeHtml(expert.name)}</h3>
                <div class="expert-stats">
                    <span>${expert.chunk_count} chunks</span>
                    <span>${expert.edge_count} edges</span>
                </div>
            `;
            card.addEventListener("click", () => showExpert(expert.slug));
            list.appendChild(card);
        }
    }

    // --- Expert Detail ---
    async function showExpert(slug) {
        const data = await fetchJSON(`/api/expert/${encodeURIComponent(slug)}`);
        if (!data) return;

        document.getElementById("experts-list").parentElement.style.display = "none";
        const detail = document.getElementById("expert-detail");
        detail.style.display = "block";

        const info = document.getElementById("expert-info");
        let connectionsHtml = "";
        if (data.connections && data.connections.length > 0) {
            const items = data.connections.map(c => `
                <li>
                    <span class="connection-type">${c.edge_type}</span>
                    ${c.direction === "outgoing" ? "&rarr;" : "&larr;"}
                    <strong>${escapeHtml(c.connected_node.name)}</strong>
                    (${c.connected_node.node_type})
                    &mdash; weight: ${c.weight.toFixed(2)}, confidence: ${c.confidence.toFixed(2)}
                </li>
            `).join("");
            connectionsHtml = `<ul class="connection-list">${items}</ul>`;
        } else {
            connectionsHtml = '<p class="no-data">No connections yet.</p>';
        }

        info.innerHTML = `
            <h2>${escapeHtml(data.name)}</h2>
            <p style="color:var(--text-secondary)">${data.chunk_count} chunks indexed</p>
            <h3 style="margin-top:1rem">Connections (${data.connections.length})</h3>
            ${connectionsHtml}
        `;
    }

    // --- Search ---
    async function doSearch() {
        const q = document.getElementById("search-input").value.trim();
        if (!q) return;

        const container = document.getElementById("search-results");
        container.innerHTML = '<p class="no-data">Searching...</p>';

        const data = await fetchJSON(`/api/search?q=${encodeURIComponent(q)}&limit=10`);
        if (!data) {
            container.innerHTML = '<p class="no-data">Search failed.</p>';
            return;
        }

        if (!data.results || data.results.length === 0) {
            container.innerHTML = '<p class="no-data">No results found.</p>';
            return;
        }

        container.innerHTML = data.results.map(r => `
            <div class="result-item">
                <span class="result-expert">${escapeHtml(r.expert || "unknown")}</span>
                <span class="result-score">score: ${(r.combined_score || r.score || 0).toFixed(3)}</span>
                <div class="result-text">${escapeHtml((r.text || "").substring(0, 300))}</div>
            </div>
        `).join("");
    }

    // --- Back button ---
    document.getElementById("back-btn").addEventListener("click", () => {
        document.getElementById("expert-detail").style.display = "none";
        document.getElementById("experts-list").parentElement.style.display = "block";
    });

    // --- Search handlers ---
    document.getElementById("search-btn").addEventListener("click", doSearch);
    document.getElementById("search-input").addEventListener("keydown", (e) => {
        if (e.key === "Enter") doSearch();
    });

    // --- SSE for live updates ---
    function connectSSE() {
        try {
            const evtSource = new EventSource(API + "/api/events");
            evtSource.addEventListener("graph_updated", () => loadDashboard());
            evtSource.addEventListener("ingest_complete", () => {
                loadDashboard();
                loadExperts();
            });
            evtSource.onerror = () => {
                console.warn("SSE connection lost, reconnecting in 5s...");
                evtSource.close();
                setTimeout(connectSSE, 5000);
            };
        } catch (e) {
            console.warn("SSE not available:", e);
        }
    }

    // --- Utilities ---
    function escapeHtml(str) {
        const div = document.createElement("div");
        div.textContent = str;
        return div.innerHTML;
    }

    // --- Init ---
    loadDashboard();
    loadExperts();
    connectSSE();
})();
