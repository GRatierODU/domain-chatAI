import asyncio
from typing import List, Dict, Optional
from datetime import datetime
import hashlib
import json
from sentence_transformers import SentenceTransformer
import logging
from ..crawler.intelligent_crawler import CrawledPage
from .multimodal_parser import MultimodalParser, ProcessedContent

# Import the centralized ChromaDB manager
try:
    from ..core.chromadb_manager import chroma_manager
except:
    # Fallback if import path is different
    import sys
    import os

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core.chromadb_manager import chroma_manager

logger = logging.getLogger(__name__)


class KnowledgeBuilder:
    """
    Builds knowledge base from processed content using centralized ChromaDB
    """

    def __init__(self, multimodal_parser: MultimodalParser):
        self.parser = multimodal_parser
        self.embedding_model = SentenceTransformer("BAAI/bge-large-en-v1.5")

    async def build_knowledge_base(self, domain: str, pages: List[CrawledPage]) -> str:
        """
        Build complete knowledge base from crawled pages
        """
        logger.info(f"Building knowledge base for {domain} with {len(pages)} pages")

        # Create collection name
        collection_name = f"website_{domain.replace('.', '_')}"

        # Use centralized manager to create collection
        collection = chroma_manager.create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine", "domain": domain}
        )

        # Process all pages
        all_chunks = []
        all_embeddings = []
        all_metadatas = []
        all_ids = []

        for page_idx, page in enumerate(pages):
            logger.info(f"Processing page {page_idx + 1}/{len(pages)}: {page.url}")

            try:
                # Process page with multimodal parser
                processed_content = await self.parser.process_page(page)

                # Convert to knowledge chunks
                chunks = self._create_knowledge_chunks(page, processed_content)

                # Add to batch
                for chunk_idx, chunk in enumerate(chunks):
                    chunk_id = f"page_{page_idx}_chunk_{chunk_idx}"
                    all_chunks.append(chunk["text"])
                    all_metadatas.append(chunk["metadata"])
                    all_ids.append(chunk_id)

            except Exception as e:
                logger.error(f"Error processing page {page.url}: {e}")
                continue

        # Generate embeddings in batches
        logger.info(f"Generating embeddings for {len(all_chunks)} chunks")
        batch_size = 32

        for i in range(0, len(all_chunks), batch_size):
            batch_texts = all_chunks[i : i + batch_size]
            batch_embeddings = self.embedding_model.encode(batch_texts)
            all_embeddings.extend(batch_embeddings.tolist())

        # Add to collection
        logger.info("Adding to ChromaDB")
        if all_chunks:  # Only add if we have data
            collection.add(
                embeddings=all_embeddings,
                documents=all_chunks,
                metadatas=all_metadatas,
                ids=all_ids,
            )

        # Save collection metadata
        self._save_collection_metadata(collection_name, domain, pages)

        logger.info(f"Knowledge base created successfully: {collection_name}")
        return collection_name

    def _create_knowledge_chunks(
        self, page: CrawledPage, content: ProcessedContent
    ) -> List[Dict]:
        """
        Convert processed content into knowledge chunks
        """
        chunks = []

        # Add page overview chunk
        overview = self._create_page_overview(page, content)
        chunks.append(overview)

        # Process text chunks
        for text_chunk in content.text_chunks:
            chunk = {
                "text": text_chunk["text"],
                "metadata": {
                    "url": page.url,
                    "page_type": page.page_type,
                    "section_type": text_chunk.get("section_type", "content"),
                    "importance": text_chunk.get("importance", 1.0),
                    "title": page.title,
                    "chunk_type": "text",
                    "crawled_at": page.crawled_at.isoformat(),
                },
            }

            # Use original text for embedding (no special embedding text)
            chunks.append(chunk)

        # Process structured data
        for data in content.structured_data:
            chunk = {
                "text": self._structured_data_to_text(data),
                "metadata": {
                    "url": page.url,
                    "page_type": page.page_type,
                    "data_type": data.get("type", "structured"),
                    "title": page.title,
                    "chunk_type": "structured_data",
                    "crawled_at": page.crawled_at.isoformat(),
                },
            }
            chunks.append(chunk)

        # Process visual elements descriptions
        for visual in content.visual_elements:
            if visual.get("caption") or visual.get("description"):
                chunk = {
                    "text": f"{visual.get('caption', '')} {visual.get('description', '')}".strip(),
                    "metadata": {
                        "url": page.url,
                        "page_type": page.page_type,
                        "visual_type": visual.get("type", "image"),
                        "title": page.title,
                        "chunk_type": "visual",
                        "crawled_at": page.crawled_at.isoformat(),
                    },
                }
                chunks.append(chunk)

        # Process interactions (forms, CTAs)
        for interaction in content.interactions:
            chunk = {
                "text": self._interaction_to_text(interaction),
                "metadata": {
                    "url": page.url,
                    "page_type": page.page_type,
                    "interaction_type": interaction.get("type", "form"),
                    "title": page.title,
                    "chunk_type": "interaction",
                    "crawled_at": page.crawled_at.isoformat(),
                },
            }
            chunks.append(chunk)

        return chunks

    def _create_page_overview(
        self, page: CrawledPage, content: ProcessedContent
    ) -> Dict:
        """
        Create an overview chunk for the page
        """
        understanding = content.page_understanding

        overview_parts = [
            f"Page: {page.title or page.url}",
            f"Type: {understanding.get('page_type', 'unknown')}",
            f"Purpose: {understanding.get('purpose', 'General information')}",
        ]

        if understanding.get("main_sections"):
            overview_parts.append(
                f"Main sections: {', '.join(understanding['main_sections'])}"
            )

        if understanding.get("key_information"):
            overview_parts.append(
                f"Key information: {understanding['key_information']}"
            )

        return {
            "text": " ".join(overview_parts),
            "metadata": {
                "url": page.url,
                "page_type": page.page_type,
                "title": page.title,
                "chunk_type": "overview",
                "importance": 2.0,  # Higher importance for overviews
                "crawled_at": page.crawled_at.isoformat(),
            },
        }

    def _structured_data_to_text(self, data: Dict) -> str:
        """
        Convert structured data to searchable text
        """
        text_parts = []

        if data.get("type") == "table":
            text_parts.append("Table data:")
            if data.get("headers"):
                text_parts.append(f"Columns: {', '.join(data['headers'])}")
            if data.get("rows"):
                for row in data["rows"][:5]:  # Limit to first 5 rows
                    text_parts.append(str(row))

        elif data.get("type") == "list":
            text_parts.append("List items:")
            for item in data.get("items", [])[:10]:  # Limit to first 10 items
                text_parts.append(str(item))

        elif data.get("type") == "json_ld":
            # Extract key business information
            if isinstance(data.get("content"), dict):
                for key in [
                    "name",
                    "description",
                    "address",
                    "telephone",
                    "email",
                    "priceRange",
                ]:
                    if key in data["content"]:
                        text_parts.append(f"{key}: {data['content'][key]}")

        else:
            text_parts.append(json.dumps(data, indent=2)[:500])  # Limit length

        return " ".join(text_parts)

    def _interaction_to_text(self, interaction: Dict) -> str:
        """
        Convert interaction element to searchable text
        """
        text_parts = []

        if interaction.get("type") == "form":
            text_parts.append(f"Form: {interaction.get('purpose', 'User input form')}")
            if interaction.get("fields"):
                field_names = [f.get("name", "unnamed") for f in interaction["fields"]]
                text_parts.append(f"Fields: {', '.join(field_names)}")

        elif interaction.get("type") == "cta":
            text_parts.append(
                f"Call to action: {interaction.get('text', 'Click here')}"
            )
            if interaction.get("action"):
                text_parts.append(f"Action: {interaction['action']}")

        return " ".join(text_parts)

    def _save_collection_metadata(
        self, collection_name: str, domain: str, pages: List[CrawledPage]
    ):
        """
        Save metadata about the collection
        """
        metadata = {
            "collection_name": collection_name,
            "domain": domain,
            "pages_count": len(pages),
            "created_at": datetime.utcnow().isoformat(),
            "page_urls": [p.url for p in pages],
            "page_types": list(set(p.page_type for p in pages)),
        }

        # Save to a metadata collection using centralized manager
        try:
            metadata_collection = chroma_manager.get_or_create_collection("_metadata")
            metadata_collection.add(
                documents=[json.dumps(metadata)],
                metadatas=[{"type": "collection_metadata", "domain": domain}],
                ids=[collection_name],
            )
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
