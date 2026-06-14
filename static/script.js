document.addEventListener("DOMContentLoaded", () => {
    let activePdf = null;
    let selectedRagImageBase64 = null;
    let selectedSumImageBase64 = null;
    let retrievedContextTexts = [];
    
    const tabLinks = document.querySelectorAll(".tab-link");
    const tabPanes = document.querySelectorAll(".tab-pane");
    
    const documentList = document.getElementById("document-list");
    const btnRefreshDocs = document.getElementById("btn-refresh-docs");
    const btnWarmup = document.getElementById("btn-warmup");
    const activeDocBanner = document.getElementById("active-doc-banner");
    
    const ragForm = document.getElementById("rag-query-form");
    const ragQueryText = document.getElementById("rag-query-text");
    const ragTopK = document.getElementById("rag-top-k");
    const ragImageInput = document.getElementById("rag-query-image");
    const ragDropzone = document.getElementById("rag-image-dropzone");
    const ragPreviewContainer = document.getElementById("rag-image-preview-container");
    const ragPreviewImg = document.getElementById("rag-image-preview");
    const btnClearRagImage = document.getElementById("btn-clear-rag-image");
    const btnSubmitRag = document.getElementById("btn-submit-rag");
    const ragSpinner = document.getElementById("rag-spinner");
    
    const ragLlmResponse = document.getElementById("rag-llm-response");
    const ragTimeStats = document.getElementById("rag-time-stats");
    const retrievedContextsList = document.getElementById("retrieved-contexts-list");
    
    const sumForm = document.getElementById("summarize-form");
    const sumInputText = document.getElementById("sum-input-text");
    const sumImageInput = document.getElementById("sum-query-image");
    const sumDropzone = document.getElementById("sum-image-dropzone");
    const sumPreviewContainer = document.getElementById("sum-image-preview-container");
    const sumPreviewImg = document.getElementById("sum-image-preview");
    const btnClearSumImage = document.getElementById("btn-clear-sum-image");
    const btnSubmitSummarize = document.getElementById("btn-submit-summarize");
    const sumSpinner = document.getElementById("sum-spinner");
    
    const summaryTextOutput = document.getElementById("summary-text-output");
    const sumTimeStats = document.getElementById("sum-time-stats");
    const metricSumTime = document.getElementById("metric-sum-time");
    const metricSumRatio = document.getElementById("metric-sum-ratio");
    const metricSumStatus = document.getElementById("metric-sum-status");
    
    const metricsModelDetails = document.getElementById("metrics-model-details");
    const valSearchLat = document.getElementById("val-search-lat");
    const valGenLat = document.getElementById("val-gen-lat");
    const valSumLat = document.getElementById("val-sum-lat");
    const valMeanSim = document.getElementById("val-mean-sim");
    const valMeanCompress = document.getElementById("val-mean-compress");
    
    const fillSearchLat = document.getElementById("fill-search-lat");
    const fillGenLat = document.getElementById("fill-gen-lat");
    const fillSumLat = document.getElementById("fill-sum-lat");
    const fillMeanSim = document.getElementById("fill-mean-sim");
    const fillMeanCompress = document.getElementById("fill-mean-compress");

    tabLinks.forEach(link => {
        link.addEventListener("click", () => {
            tabLinks.forEach(l => l.classList.remove("active"));
            tabPanes.forEach(p => p.classList.remove("active"));
            
            link.classList.add("active");
            const tabId = link.getAttribute("data-tab");
            document.getElementById(tabId).classList.add("active");
            
            if (tabId === "tab-metrics") {
                loadMetrics();
            }
        });
    });

    async function loadDocuments() {
        try {
            const res = await fetch("/api/list_documents");
            const data = await res.json();
            
            documentList.innerHTML = "";
            let hasIndexing = false;
            
            if (data.length === 0) {
                documentList.innerHTML = `<div class="loading-spinner">No PDFs found in data/ directory.</div>`;
                return;
            }
            
            data.forEach(doc => {
                const card = document.createElement("div");
                card.className = `doc-card ${activePdf === doc.filename ? "active" : ""}`;
                card.setAttribute("data-filename", doc.filename);
                
                let tagClass = "not-indexed";
                if (doc.status === "Indexed") tagClass = "indexed";
                else if (doc.status === "Indexing") {
                    tagClass = "indexing";
                    hasIndexing = true;
                } else if (doc.status === "Failed") tagClass = "failed";
                
                const metaText = doc.status === "Indexed" 
                    ? `${doc.embeddings_count} elements` 
                    : doc.progress || "Not started";
                    
                card.innerHTML = `
                    <div class="doc-name">${doc.filename}</div>
                    <div class="doc-meta">
                        <span class="doc-status-tag ${tagClass}">${doc.status}</span>
                        <span class="doc-embeddings-count">${metaText}</span>
                    </div>
                `;
                
                if (doc.status === "Not Indexed" || doc.status === "Failed") {
                    const btn = document.createElement("button");
                    btn.className = "btn-index-trigger";
                    btn.innerText = "Index Document";
                    btn.addEventListener("click", (e) => {
                        e.stopPropagation();
                        startIndexing(doc.filename);
                    });
                    card.appendChild(btn);
                }
                
                card.addEventListener("click", () => {
                    if (doc.status === "Indexed") {
                        selectPdf(doc.filename);
                    } else {
                        alert(`Please index this document first before selecting it.`);
                    }
                });
                
                documentList.appendChild(card);
            });
            
            if (hasIndexing) {
                setTimeout(loadDocuments, 2500);
            }
        } catch (e) {
            console.error("Failed to load documents:", e);
            documentList.innerHTML = `<div class="loading-spinner">Error loading document list.</div>`;
        }
    }
    
    async function startIndexing(filename) {
        try {
            const res = await fetch("/api/index_document", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ filename })
            });
            const data = await res.json();
            loadDocuments();
        } catch (e) {
            console.error("Indexing request failed:", e);
            alert("Failed to start indexing job.");
        }
    }
    
    function selectPdf(filename) {
        activePdf = filename;
        document.querySelectorAll(".doc-card").forEach(card => {
            if (card.getAttribute("data-filename") === filename) {
                card.classList.add("active");
            } else {
                card.classList.remove("active");
            }
        });
        
        activeDocBanner.className = "active-doc-banner";
        activeDocBanner.innerHTML = `
            <span style="font-weight:700;margin-right:8px;">Active PDF:</span>
            <span>${filename}</span>
        `;
    }

    btnRefreshDocs.addEventListener("click", loadDocuments);

    btnWarmup.addEventListener("click", async () => {
        btnWarmup.disabled = true;
        btnWarmup.innerText = "Warming up...";
        try {
            const res = await fetch("/api/warmup");
            const data = await res.json();
            alert("Models loaded and warmed up successfully!");
        } catch (e) {
            console.error(e);
            alert("Model warmup encountered an issue.");
        } finally {
            btnWarmup.disabled = false;
            btnWarmup.innerText = "Warmup Models";
        }
    });

    function setupDropzone(dropzone, fileInput, previewContainer, previewImg, clearBtn, onImageLoaded) {
        dropzone.addEventListener("click", () => fileInput.click());
        
        fileInput.addEventListener("change", (e) => {
            const file = e.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = (event) => {
                    const base64 = event.target.result;
                    previewImg.src = base64;
                    previewContainer.classList.remove("hidden");
                    dropzone.querySelector(".upload-placeholder").classList.add("hidden");
                    onImageLoaded(base64);
                };
                reader.readAsDataURL(file);
            }
        });
        
        clearBtn.addEventListener("click", (e) => {
            e.stopPropagation();
            fileInput.value = "";
            previewImg.src = "";
            previewContainer.classList.add("hidden");
            dropzone.querySelector(".upload-placeholder").classList.remove("hidden");
            onImageLoaded(null);
        });
    }

    setupDropzone(
        ragDropzone, 
        ragImageInput, 
        ragPreviewContainer, 
        ragPreviewImg, 
        btnClearRagImage, 
        (base64) => { selectedRagImageBase64 = base64; }
    );

    setupDropzone(
        sumDropzone, 
        sumImageInput, 
        sumPreviewContainer, 
        sumPreviewImg, 
        btnClearSumImage, 
        (base64) => { selectedSumImageBase64 = base64; }
    );

    btnSubmitRag.addEventListener("click", async () => {
        const textQuery = ragQueryText.value.strip ? ragQueryText.value.strip() : ragQueryText.value.trim();
        const topK = parseInt(ragTopK.value);
        
        if (!activePdf) {
            alert("Please select and index a PDF in the sidebar before searching.");
            return;
        }
        
        if (!textQuery && !selectedRagImageBase64) {
            alert("Please enter a question or upload a query image.");
            return;
        }
        
        btnSubmitRag.disabled = true;
        ragSpinner.classList.remove("hidden");
        ragLlmResponse.className = "response-placeholder";
        ragLlmResponse.innerText = "Retrieving context and generating answer locally on CPU...";
        ragTimeStats.innerText = "Running...";
        retrievedContextsList.innerHTML = `<div class="loading-spinner">Searching FAISS...</div>`;
        
        try {
            const searchRes = await fetch("/api/search", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    pdf_name: activePdf,
                    query_text: textQuery || null,
                    query_image_base64: selectedRagImageBase64 || null,
                    top_k: topK
                })
            });
            
            if (!searchRes.ok) {
                const err = await searchRes.json();
                throw new Error(err.detail || "Search failed");
            }
            
            const searchData = await searchRes.json();
            const results = searchData.results;
            
            retrievedContextsList.innerHTML = "";
            retrievedContextTexts = [];
            
            if (results.length === 0) {
                retrievedContextsList.innerHTML = `<div class="no-results-placeholder">No matching content found.</div>`;
            } else {
                results.forEach((res, i) => {
                    retrievedContextTexts.push(res.content);
                    
                    const item = document.createElement("div");
                    item.className = "context-item";
                    
                    const scorePercent = (res.score * 100).toFixed(1);
                    const isImage = res.image_path !== null;
                    
                    let bodyHtml = "";
                    if (isImage) {
                        bodyHtml = `
                            <div class="context-body-layout">
                                <div class="context-content">${res.content}</div>
                                <div class="context-img-container">
                                    <img src="${res.image_path}" alt="Retrieved Image">
                                </div>
                            </div>
                        `;
                    } else {
                        bodyHtml = `
                            <div class="context-body-layout text-only">
                                <div class="context-content">${res.content}</div>
                            </div>
                        `;
                    }
                    
                    item.innerHTML = `
                        <div class="context-header">
                            <div class="context-meta-tags">
                                <span class="meta-tag page">Page ${res.page}</span>
                                <span class="meta-tag type">${res.type === 'image_visual' || res.type === 'image_caption' ? 'Image' : 'Text'}</span>
                                <span class="meta-tag similarity">Similarity: ${scorePercent}%</span>
                            </div>
                            <button class="btn-use-context" data-index="${i}">Send to Summarizer</button>
                        </div>
                        ${bodyHtml}
                    `;
                    
                    item.querySelector(".btn-use-context").addEventListener("click", (e) => {
                        const idx = parseInt(e.target.getAttribute("data-index"));
                        sendToSummarizer(retrievedContextTexts[idx], isImage ? res.image_path : null);
                    });
                    
                    retrievedContextsList.appendChild(item);
                });
            }
            
            const ragRes = await fetch("/api/rag", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    query_text: textQuery || null,
                    query_image_base64: selectedRagImageBase64 || null,
                    retrieved_contexts: retrievedContextTexts
                })
            });
            
            const ragData = await ragRes.json();
            
            ragLlmResponse.className = "response-text";
            ragLlmResponse.innerText = ragData.response;
            ragLlmResponse.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            
            const totalTime = (searchData.search_time_sec + ragData.generation_time_sec).toFixed(2);
            ragTimeStats.innerText = `Search: ${searchData.search_time_sec.toFixed(2)}s | Gen: ${ragData.generation_time_sec.toFixed(2)}s | Total: ${totalTime}s`;
            
        } catch (e) {
            console.error(e);
            ragLlmResponse.className = "response-placeholder";
            ragLlmResponse.innerText = `Error: ${e.message}`;
            ragTimeStats.innerText = "Error";
        } finally {
            btnSubmitRag.disabled = false;
            ragSpinner.classList.add("hidden");
        }
    });

    function sendToSummarizer(text, imagePath) {
        sumInputText.value = text;
        
        if (imagePath) {
            sumPreviewImg.src = imagePath;
            sumPreviewContainer.classList.remove("hidden");
            sumDropzone.querySelector(".upload-placeholder").classList.add("hidden");
            
            fetch(imagePath)
                .then(res => res.blob())
                .then(blob => {
                    const reader = new FileReader();
                    reader.onload = (e) => {
                        selectedSumImageBase64 = e.target.result;
                    };
                    reader.readAsDataURL(blob);
                });
        } else {
            sumImageInput.value = "";
            sumPreviewImg.src = "";
            sumPreviewContainer.classList.add("hidden");
            sumDropzone.querySelector(".upload-placeholder").classList.remove("hidden");
            selectedSumImageBase64 = null;
        }
        
        document.getElementById("nav-tab-summarizer").click();
    }

    btnSubmitSummarize.addEventListener("click", async () => {
        const text = sumInputText.value.trim();
        
        if (!text && !selectedSumImageBase64) {
            alert("Please provide some text or upload an image to summarize.");
            return;
        }
        
        btnSubmitSummarize.disabled = true;
        sumSpinner.classList.remove("hidden");
        summaryTextOutput.className = "summary-placeholder";
        summaryTextOutput.innerText = "Summarizing context using local LLM (takes approx. 5-15s)...";
        sumTimeStats.innerText = "Running...";
        
        metricSumTime.innerText = "--";
        metricSumRatio.innerText = "--";
        metricSumStatus.innerText = "Running...";
        metricSumStatus.className = "metric-value highlight-cyan";
        
        try {
            const res = await fetch("/api/summarize", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    text: text || null,
                    image_base64: selectedSumImageBase64 || null
                })
            });
            
            if (!res.ok) {
                const err = await res.json();
                throw new Error(err.detail || "Summarization failed");
            }
            
            const data = await res.json();
            
            summaryTextOutput.className = "summary-text";
            summaryTextOutput.innerText = data.summary;
            
            sumTimeStats.innerText = `Completed in ${data.time_taken_sec.toFixed(2)}s`;
            metricSumTime.innerText = `${data.time_taken_sec.toFixed(2)}s`;
            
            const compressPercent = (data.compression_ratio * 100).toFixed(1);
            metricSumRatio.innerText = text ? `${compressPercent}%` : "N/A (Image)";
            
            metricSumStatus.innerText = "Completed";
            metricSumStatus.className = "metric-value highlight-green";
        } catch (e) {
            console.error(e);
            summaryTextOutput.className = "summary-placeholder";
            summaryTextOutput.innerText = `Error: ${e.message}`;
            sumTimeStats.innerText = "Error";
            
            metricSumStatus.innerText = "Failed";
            metricSumStatus.className = "metric-value highlight-red";
        } finally {
            btnSubmitSummarize.disabled = false;
            sumSpinner.classList.add("hidden");
        }
    });

    async function loadMetrics() {
        try {
            const res = await fetch("/api/metrics");
            const data = await res.json();
            
            const details = data.model_details;
            metricsModelDetails.innerHTML = `
                <div class="spec-item">
                    <strong>Embedding Engine:</strong>
                    <span>${details.embedding_model}</span>
                </div>
                <div class="spec-item">
                    <strong>Captioning Engine:</strong>
                    <span>${details.captioning_model}</span>
                </div>
                <div class="spec-item">
                    <strong>Causal LLM Engine:</strong>
                    <span>${details.llm_model}</span>
                </div>
                <div class="spec-item">
                    <strong>Hardware Context:</strong>
                    <span class="highlight-cyan">${details.hardware_accelerator}</span>
                </div>
            `;
            
            const averages = data.averages;
            
            valSearchLat.innerText = `${averages.search_latency_sec.toFixed(2)}s`;
            const searchPct = Math.min((averages.search_latency_sec / 1.0) * 100, 100);
            fillSearchLat.style.width = `${searchPct}%`;
            
            valGenLat.innerText = `${averages.generation_latency_sec.toFixed(2)}s`;
            const genPct = Math.min((averages.generation_latency_sec / 10.0) * 100, 100);
            fillGenLat.style.width = `${genPct}%`;
            
            valSumLat.innerText = `${averages.summary_latency_sec.toFixed(2)}s`;
            const sumPct = Math.min((averages.summary_latency_sec / 15.0) * 100, 100);
            fillSumLat.style.width = `${sumPct}%`;
            
            valMeanSim.innerText = `${(averages.retrieval_similarity * 100).toFixed(1)}%`;
            fillMeanSim.style.width = `${averages.retrieval_similarity * 100}%`;
            
            valMeanCompress.innerText = `${(averages.compression_ratio * 100).toFixed(1)}%`;
            fillMeanCompress.style.width = `${averages.compression_ratio * 100}%`;
            
        } catch (e) {
            console.error("Failed to load metrics:", e);
        }
    }

    loadDocuments();
});
