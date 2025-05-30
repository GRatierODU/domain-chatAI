import torch
from PIL import Image
from typing import Dict, List, Optional, Tuple
import numpy as np
from transformers import (
    AutoProcessor,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
import io
import json
import logging
from dataclasses import dataclass
from bs4 import BeautifulSoup
import re

# Import required classes
from ..crawler.intelligent_crawler import CrawledPage
from .visual_understanding import VisualAnalyzer
from .layout_analyzer import LayoutAnalyzer

logger = logging.getLogger(__name__)


@dataclass
class ProcessedContent:
    text_chunks: List[Dict]
    visual_elements: List[Dict]
    structured_data: List[Dict]
    interactions: List[Dict]
    relationships: List[Dict]
    page_understanding: Dict


class MultimodalParser:
    """
    Production-ready multimodal parser for complete website understanding
    """

    def __init__(self, model_config: Dict):
        self.config = model_config
        self._load_models()

    def _load_models(self):
        """
        Load all required models with optimization
        """
        logger.info("Loading multimodal models...")

        try:
            # For now, we'll use simpler models that work without large downloads
            # In production, you'd use the full vision-language models

            # Layout understanding
            self.layout_analyzer = LayoutAnalyzer()

            # Visual content analyzer
            self.visual_analyzer = VisualAnalyzer()

            logger.info("✓ Models loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            # Continue with limited functionality
            self.layout_analyzer = None
            self.visual_analyzer = None

    async def process_page(self, crawled_page: CrawledPage) -> ProcessedContent:
        """
        Process a crawled page with full multimodal understanding
        """
        logger.info(f"Processing {crawled_page.url}")

        # 1. Overall page understanding using screenshots
        page_understanding = await self._understand_page_context(crawled_page)

        # 2. Layout analysis
        layout_structure = {}
        if self.layout_analyzer and crawled_page.screenshots:
            layout_structure = await self.layout_analyzer.analyze(
                crawled_page.screenshots[0] if crawled_page.screenshots else None,
                crawled_page.html,
            )

        # 3. Extract and process text with context
        text_chunks = await self._process_text_content(
            crawled_page, layout_structure, page_understanding
        )

        # 4. Process visual elements
        visual_elements = await self._process_visual_content(
            crawled_page, page_understanding
        )

        # 5. Process structured data (tables, lists, etc.)
        structured_data = await self._process_structured_content(
            crawled_page, page_understanding
        )

        # 6. Process interactive elements
        interactions = await self._process_interactions(
            crawled_page, page_understanding
        )

        # 7. Identify relationships between elements
        relationships = self._identify_relationships(
            text_chunks, visual_elements, structured_data, interactions
        )

        return ProcessedContent(
            text_chunks=text_chunks,
            visual_elements=visual_elements,
            structured_data=structured_data,
            interactions=interactions,
            relationships=relationships,
            page_understanding=page_understanding,
        )

    async def _understand_page_context(self, page: CrawledPage) -> Dict:
        """
        Understand overall page context and purpose
        """
        # For now, use HTML analysis instead of vision model
        soup = BeautifulSoup(page.html, "html.parser")

        understanding = {
            "page_type": self._detect_page_type(page, soup),
            "purpose": self._extract_page_purpose(soup),
            "main_sections": self._identify_main_sections(soup),
            "key_information": self._extract_key_info(page, soup),
        }

        return understanding

    async def _process_text_content(
        self, page: CrawledPage, layout: Dict, context: Dict
    ) -> List[Dict]:
        """
        Process text content with semantic understanding and context
        """
        soup = BeautifulSoup(page.html, "html.parser")

        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()

        text_chunks = []

        # Process main content areas
        main_content_selectors = [
            "main",
            "article",
            '[role="main"]',
            ".content",
            "#content",
        ]
        for selector in main_content_selectors:
            elements = soup.select(selector)
            for element in elements:
                text = element.get_text(separator=" ", strip=True)
                if text and len(text) > 50:  # Minimum text length
                    chunk = {
                        "text": text,
                        "section_type": "main_content",
                        "importance": 2.0,
                        "metadata": {
                            "url": page.url,
                            "page_type": context.get("page_type"),
                            "selector": selector,
                        },
                        "embedding_text": self._create_contextual_embedding(
                            text, {"type": "main_content"}, context, page
                        ),
                    }
                    text_chunks.append(chunk)

        # Process other sections
        for section in soup.find_all(["section", "div", "aside"]):
            text = section.get_text(separator=" ", strip=True)
            if text and len(text) > 30:
                section_type = self._determine_section_type(section)
                chunk = {
                    "text": text,
                    "section_type": section_type,
                    "importance": 1.0,
                    "metadata": {
                        "url": page.url,
                        "page_type": context.get("page_type"),
                        "section_id": section.get("id", ""),
                        "section_class": " ".join(section.get("class", [])),
                    },
                    "embedding_text": self._create_contextual_embedding(
                        text, {"type": section_type}, context, page
                    ),
                }
                text_chunks.append(chunk)

        return text_chunks

    async def _process_visual_content(
        self, page: CrawledPage, context: Dict
    ) -> List[Dict]:
        """
        Process visual elements from the page
        """
        visual_elements = []

        # Process images from media
        for media_item in page.media:
            if media_item.get("type") == "image":
                element = {
                    "type": "image",
                    "src": media_item.get("src", ""),
                    "alt": media_item.get("alt", ""),
                    "title": media_item.get("title", ""),
                    "context": context.get("page_type"),
                    "description": media_item.get("alt", "")
                    or media_item.get("title", ""),
                }
                visual_elements.append(element)

        return visual_elements

    async def _process_structured_content(
        self, page: CrawledPage, context: Dict
    ) -> List[Dict]:
        """
        Process structured data like tables, lists, etc.
        """
        structured_data = []
        soup = BeautifulSoup(page.html, "html.parser")

        # Process tables
        for table in soup.find_all("table"):
            table_data = self._extract_table_data(table)
            if table_data:
                structured_data.append(
                    {
                        "type": "table",
                        "data": table_data,
                        "context": context.get("page_type"),
                    }
                )

        # Process definition lists
        for dl in soup.find_all("dl"):
            dl_data = self._extract_dl_data(dl)
            if dl_data:
                structured_data.append(
                    {
                        "type": "definition_list",
                        "data": dl_data,
                        "context": context.get("page_type"),
                    }
                )

        # Add JSON-LD structured data
        if page.structured_data.get("json_ld"):
            for item in page.structured_data["json_ld"]:
                structured_data.append(
                    {
                        "type": "json_ld",
                        "data": item,
                        "context": context.get("page_type"),
                    }
                )

        return structured_data

    async def _process_interactions(
        self, page: CrawledPage, context: Dict
    ) -> List[Dict]:
        """
        Process interactive elements like forms, buttons, etc.
        """
        interactions = []

        # Process forms
        for form in page.forms:
            interaction = {
                "type": "form",
                "action": form.get("action", ""),
                "method": form.get("method", "get"),
                "purpose": self._determine_form_purpose(form),
                "fields": form.get("inputs", []),
                "context": context.get("page_type"),
            }
            interactions.append(interaction)

        return interactions

    def _create_contextual_embedding(
        self, text: str, section: Dict, context: Dict, page: CrawledPage
    ) -> str:
        """
        Create rich embedding text with full context
        """
        parts = [
            f"On the {context.get('page_type', 'webpage')} page",
            f"of {page.url.split('/')[2]}",  # domain
            f"in the {section.get('type', 'main')} section",
            text[:500],  # Limit text length
        ]

        return " ".join(parts)

    def _detect_page_type(self, page: CrawledPage, soup: BeautifulSoup) -> str:
        """
        Detect the type of page
        """
        # Check URL patterns
        url_lower = page.url.lower()
        if "product" in url_lower or "item" in url_lower:
            return "product"
        elif "about" in url_lower:
            return "about"
        elif "contact" in url_lower:
            return "contact"
        elif "blog" in url_lower or "article" in url_lower:
            return "blog"
        elif "cart" in url_lower or "checkout" in url_lower:
            return "checkout"

        # Check page title
        if page.title:
            title_lower = page.title.lower()
            if "about" in title_lower:
                return "about"
            elif "contact" in title_lower:
                return "contact"
            elif "product" in title_lower:
                return "product"

        # Check structured data
        if page.structured_data.get("json_ld"):
            for item in page.structured_data["json_ld"]:
                if isinstance(item, dict):
                    item_type = item.get("@type", "").lower()
                    if "product" in item_type:
                        return "product"
                    elif "article" in item_type:
                        return "blog"
                    elif "organization" in item_type:
                        return "about"

        return "general"

    def _extract_page_purpose(self, soup: BeautifulSoup) -> str:
        """
        Extract the main purpose of the page
        """
        # Check meta description
        meta_desc = soup.find("meta", attrs={"name": "description"})
        if meta_desc:
            return meta_desc.get("content", "")[:200]

        # Check first paragraph
        first_p = soup.find("p")
        if first_p:
            return first_p.get_text(strip=True)[:200]

        return "General information page"

    def _identify_main_sections(self, soup: BeautifulSoup) -> List[str]:
        """
        Identify main sections of the page
        """
        sections = []

        # Look for headings
        for heading in soup.find_all(["h1", "h2", "h3"])[:10]:  # Limit to first 10
            text = heading.get_text(strip=True)
            if text and len(text) < 100:
                sections.append(text)

        return sections

    def _extract_key_info(self, page: CrawledPage, soup: BeautifulSoup) -> str:
        """
        Extract key information from the page
        """
        key_info = []

        # Check for contact info
        email_pattern = r"[\w\.-]+@[\w\.-]+\.\w+"
        emails = re.findall(email_pattern, str(soup))
        if emails:
            key_info.append(f"Email: {emails[0]}")

        # Check for phone numbers
        phone_pattern = r"[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]"
        phones = re.findall(phone_pattern, str(soup))
        if phones:
            key_info.append(f"Phone: {phones[0]}")

        # Check for addresses in structured data
        if page.structured_data.get("json_ld"):
            for item in page.structured_data["json_ld"]:
                if isinstance(item, dict) and "address" in item:
                    key_info.append(f"Address: {item['address']}")

        return " | ".join(key_info) if key_info else ""

    def _determine_section_type(self, element) -> str:
        """
        Determine the type of section based on classes and content
        """
        classes = " ".join(element.get("class", [])).lower()
        element_id = (element.get("id", "") or "").lower()

        if "nav" in classes or "nav" in element_id:
            return "navigation"
        elif "footer" in classes or "footer" in element_id:
            return "footer"
        elif "header" in classes or "header" in element_id:
            return "header"
        elif "sidebar" in classes or "aside" in element.name:
            return "sidebar"
        elif "product" in classes or "product" in element_id:
            return "product_info"
        elif "price" in classes or "price" in element_id:
            return "pricing"

        return "content"

    def _extract_table_data(self, table) -> Dict:
        """
        Extract data from HTML table
        """
        headers = []
        rows = []

        # Extract headers
        for th in table.find_all("th"):
            headers.append(th.get_text(strip=True))

        # Extract rows
        for tr in table.find_all("tr"):
            cells = tr.find_all(["td", "th"])
            if cells:
                row_data = [cell.get_text(strip=True) for cell in cells]
                if any(row_data):  # Skip empty rows
                    rows.append(row_data)

        if headers or rows:
            return {"headers": headers, "rows": rows}
        return None

    def _extract_dl_data(self, dl) -> List[Dict]:
        """
        Extract data from definition list
        """
        items = []
        current_dt = None

        for child in dl.children:
            if child.name == "dt":
                current_dt = child.get_text(strip=True)
            elif child.name == "dd" and current_dt:
                items.append(
                    {"term": current_dt, "definition": child.get_text(strip=True)}
                )

        return items

    def _determine_form_purpose(self, form: Dict) -> str:
        """
        Determine the purpose of a form
        """
        action = form.get("action", "").lower()
        form_id = form.get("id", "").lower()
        form_class = form.get("class", "").lower()

        # Check action URL
        if "search" in action:
            return "search"
        elif "contact" in action or "contact" in form_id or "contact" in form_class:
            return "contact"
        elif "subscribe" in action or "newsletter" in action:
            return "newsletter"
        elif "login" in action or "signin" in action:
            return "login"
        elif "register" in action or "signup" in action:
            return "registration"
        elif "checkout" in action or "cart" in action:
            return "checkout"

        # Check input types
        inputs = form.get("inputs", [])
        has_email = any(inp.get("type") == "email" for inp in inputs)
        has_password = any(inp.get("type") == "password" for inp in inputs)

        if has_email and has_password:
            return "login"
        elif has_email:
            return "contact"

        return "general"

    def _identify_relationships(
        self,
        text_chunks: List[Dict],
        visual_elements: List[Dict],
        structured_data: List[Dict],
        interactions: List[Dict],
    ) -> List[Dict]:
        """
        Identify relationships between different elements
        """
        relationships = []

        # For now, return empty list
        # In production, this would analyze relationships between elements

        return relationships
