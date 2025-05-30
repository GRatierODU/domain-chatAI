import asyncio
from typing import List, Dict, Set, Optional, Tuple
from urllib.parse import urljoin, urlparse, parse_qs
import aiohttp
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright, Page
import hashlib
from datetime import datetime
import re
import json
from dataclasses import dataclass, field
import logging
from .discovery_strategies import DiscoveryStrategies

logger = logging.getLogger(__name__)


@dataclass
class CrawledPage:
    url: str
    title: str
    content: str
    html: str
    screenshots: List[bytes] = field(default_factory=list)
    structured_data: Dict = field(default_factory=dict)
    meta_data: Dict = field(default_factory=dict)
    links: List[str] = field(default_factory=list)
    forms: List[Dict] = field(default_factory=list)
    media: List[Dict] = field(default_factory=list)
    content_hash: str = ""
    crawled_at: datetime = field(default_factory=datetime.utcnow)
    page_type: str = "general"
    importance_score: float = 1.0


class IntelligentCrawler:
    """
    Production-ready crawler with advanced discovery and understanding
    """

    def __init__(self, domain: str, max_pages: int = 1000):
        self.domain = domain.replace("https://", "").replace("http://", "")
        self.base_url = f"https://{self.domain}"
        self.max_pages = max_pages

        # Crawl state
        self.visited_urls: Set[str] = set()
        self.to_visit: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self.pages: List[CrawledPage] = []
        self.failed_urls: Dict[str, str] = {}

        # Discovery strategies
        self.discovery = DiscoveryStrategies(self.domain)

        # URL patterns for different page types
        self.url_patterns = {
            "product": [r"/product/", r"/item/", r"/p/\d+"],
            "category": [r"/category/", r"/collections/", r"/c/"],
            "blog": [r"/blog/", r"/news/", r"/articles/"],
            "about": [r"/about", r"/team", r"/company"],
            "contact": [r"/contact", r"/support"],
            "legal": [r"/privacy", r"/terms", r"/legal"],
            "pricing": [r"/pricing", r"/plans", r"/subscribe"],
        }

    async def start(self) -> List[CrawledPage]:
        """
        Start intelligent crawling process
        """
        logger.info(f"Starting crawl of {self.domain}")

        # Phase 1: Discovery - Find all possible URLs
        discovered_urls = await self._discovery_phase()
        logger.info(f"Discovered {len(discovered_urls)} potential URLs")

        # Phase 2: Prioritization - Sort by importance
        prioritized_urls = await self._prioritize_urls(discovered_urls)

        # Add to queue with priority
        for priority, url in prioritized_urls:
            await self.to_visit.put((priority, url))

        # Phase 3: Crawling - Extract content with visual understanding
        await self._crawl_phase()

        logger.info(f"Crawl complete. Processed {len(self.pages)} pages")
        return self.pages

    async def _discovery_phase(self) -> Set[str]:
        """
        Discover all possible URLs using multiple strategies
        """
        all_urls = set()

        # 1. Standard discovery methods
        all_urls.update(await self.discovery.discover_from_sitemap())
        all_urls.update(await self.discovery.discover_from_robots())

        # 2. Homepage deep scan
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)  # HEADLESS MODE
            page = await browser.new_page()

            try:
                await page.goto(self.base_url, wait_until="networkidle")

                # Extract all links from navigation, footer, etc
                nav_links = await self._extract_navigation_links(page)
                all_urls.update(nav_links)

                # JavaScript-rendered links
                js_links = await self._extract_javascript_links(page)
                all_urls.update(js_links)

                # Search functionality discovery
                search_urls = await self.discovery.discover_via_search(page)
                all_urls.update(search_urls)

            except Exception as e:
                logger.error(f"Discovery phase error: {e}")
            finally:
                await browser.close()

        # 3. External discovery
        all_urls.update(await self.discovery.discover_via_search_engines())

        # 4. Pattern-based URL generation
        all_urls.update(self._generate_common_urls())

        # Filter to same domain only
        return {url for url in all_urls if self._is_same_domain(url)}

    async def _crawl_phase(self):
        """
        Main crawling phase with concurrent workers
        """
        # Create worker pool
        workers = []
        for i in range(min(5, self.max_pages // 100 + 1)):  # Scale workers
            workers.append(asyncio.create_task(self._crawler_worker(i)))

        # Wait for all workers
        await asyncio.gather(*workers)

    async def _crawler_worker(self, worker_id: int):
        """
        Individual crawler worker with visual understanding
        """
        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=True,  # HEADLESS MODE
                args=["--disable-blink-features=AutomationControlled"],
            )

            context = await browser.new_context(
                viewport={"width": 1920, "height": 1080},
                device_scale_factor=1.5,
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            )

            page = await context.new_page()

            # Enable request interception for efficiency
            await page.route(
                "**/*.{png,jpg,jpeg,gif,svg,mp3,mp4,avi,flac,ogg,wav,webm}",
                lambda route: route.abort(),
            )

            while not self.to_visit.empty() and len(self.visited_urls) < self.max_pages:
                try:
                    priority, url = await self.to_visit.get()

                    if url in self.visited_urls:
                        continue

                    logger.info(f"Worker {worker_id}: Crawling {url}")

                    page_data = await self._crawl_page_complete(page, url)

                    if page_data:
                        self.pages.append(page_data)
                        self.visited_urls.add(url)

                        # Add new URLs to queue
                        for link in page_data.links:
                            if link not in self.visited_urls and self._should_crawl(
                                link
                            ):
                                link_priority = self._calculate_url_priority(link)
                                await self.to_visit.put((link_priority, link))

                except asyncio.TimeoutError:
                    logger.error(f"Timeout crawling {url}")
                    self.failed_urls[url] = "timeout"
                except Exception as e:
                    logger.error(f"Error crawling {url}: {e}")
                    self.failed_urls[url] = str(e)

            await browser.close()

    async def _crawl_page_complete(self, page: Page, url: str) -> Optional[CrawledPage]:
        """
        Crawl page with complete visual and structural understanding
        """
        try:
            # Navigate with network idle
            response = await page.goto(url, wait_until="networkidle", timeout=30000)

            if not response or response.status >= 400:
                return None

            # Wait for dynamic content
            await page.wait_for_timeout(2000)

            # Scroll to load lazy content
            await self._scroll_page(page)

            # Extract everything
            crawled_page = CrawledPage(
                url=url,
                title=await page.title() or "",
                content="",  # Will be filled by processor
                html=await page.content(),
            )

            # Take screenshots for visual understanding
            crawled_page.screenshots = await self._capture_page_screenshots(page)

            # Extract structured data
            crawled_page.structured_data = await self._extract_structured_data(page)

            # Extract metadata
            crawled_page.meta_data = await self._extract_metadata(page)

            # Extract all links
            crawled_page.links = await self._extract_all_links(page)

            # Extract forms
            crawled_page.forms = await self._extract_forms(page)

            # Extract media
            crawled_page.media = await self._extract_media(page)

            # Determine page type
            crawled_page.page_type = self._determine_page_type(url, crawled_page)

            # Calculate importance
            crawled_page.importance_score = self._calculate_importance(crawled_page)

            # Generate content hash
            crawled_page.content_hash = hashlib.sha256(
                crawled_page.html.encode()
            ).hexdigest()

            return crawled_page

        except Exception as e:
            logger.error(f"Error in complete crawl of {url}: {e}")
            return None

    async def _capture_page_screenshots(self, page: Page) -> List[bytes]:
        """
        Capture multiple screenshots for visual understanding
        """
        screenshots = []

        try:
            # Full page screenshot
            full_page = await page.screenshot(full_page=True)
            screenshots.append(full_page)

            # Above the fold
            viewport = await page.screenshot()
            screenshots.append(viewport)

            # Key sections if identifiable
            sections = await page.query_selector_all(
                "main, article, .content, #content"
            )
            for section in sections[:3]:  # Limit to 3 sections
                try:
                    screenshot = await section.screenshot()
                    screenshots.append(screenshot)
                except:
                    pass
        except Exception as e:
            logger.error(f"Error capturing screenshots: {e}")

        return screenshots

    async def _extract_structured_data(self, page: Page) -> Dict:
        """
        Extract all structured data from the page
        """
        structured_data = {}

        try:
            # JSON-LD
            json_ld = await page.evaluate(
                """
                () => {
                    const scripts = document.querySelectorAll('script[type="application/ld+json"]');
                    return Array.from(scripts).map(s => {
                        try {
                            return JSON.parse(s.textContent);
                        } catch {
                            return null;
                        }
                    }).filter(Boolean);
                }
            """
            )
            if json_ld:
                structured_data["json_ld"] = json_ld

            # Open Graph
            og_data = await page.evaluate(
                """
                () => {
                    const metas = document.querySelectorAll('meta[property^="og:"]');
                    const data = {};
                    metas.forEach(meta => {
                        data[meta.getAttribute('property')] = meta.getAttribute('content');
                    });
                    return data;
                }
            """
            )
            if og_data:
                structured_data["open_graph"] = og_data

            # Microdata
            microdata = await page.evaluate(
                """
                () => {
                    const items = document.querySelectorAll('[itemscope]');
                    return Array.from(items).map(item => ({
                        type: item.getAttribute('itemtype'),
                        properties: Array.from(item.querySelectorAll('[itemprop]')).map(prop => ({
                            name: prop.getAttribute('itemprop'),
                            content: prop.textContent || prop.getAttribute('content')
                        }))
                    }));
                }
            """
            )
            if microdata:
                structured_data["microdata"] = microdata

        except Exception as e:
            logger.error(f"Error extracting structured data: {e}")

        return structured_data

    async def _extract_metadata(self, page: Page) -> Dict:
        """
        Extract page metadata
        """
        metadata = {}

        try:
            metadata = await page.evaluate(
                """
                () => {
                    const meta = {};
                    
                    // Basic meta tags
                    const description = document.querySelector('meta[name="description"]');
                    if (description) meta.description = description.content;
                    
                    const keywords = document.querySelector('meta[name="keywords"]');
                    if (keywords) meta.keywords = keywords.content;
                    
                    const author = document.querySelector('meta[name="author"]');
                    if (author) meta.author = author.content;
                    
                    const robots = document.querySelector('meta[name="robots"]');
                    if (robots) meta.robots = robots.content;
                    
                    // Language
                    meta.language = document.documentElement.lang || 'en';
                    
                    // Canonical URL
                    const canonical = document.querySelector('link[rel="canonical"]');
                    if (canonical) meta.canonical = canonical.href;
                    
                    return meta;
                }
            """
            )
        except Exception as e:
            logger.error(f"Error extracting metadata: {e}")

        return metadata

    async def _extract_all_links(self, page: Page) -> List[str]:
        """
        Extract all links from the page
        """
        links = []

        try:
            raw_links = await page.evaluate(
                """
                () => {
                    const links = new Set();
                    document.querySelectorAll('a[href]').forEach(a => {
                        links.add(a.href);
                    });
                    return Array.from(links);
                }
            """
            )

            # Filter and clean links
            for link in raw_links:
                if self._is_valid_url(link):
                    links.append(link)

        except Exception as e:
            logger.error(f"Error extracting links: {e}")

        return links

    async def _extract_forms(self, page: Page) -> List[Dict]:
        """
        Extract form information
        """
        forms = []

        try:
            forms = await page.evaluate(
                """
                () => {
                    return Array.from(document.querySelectorAll('form')).map(form => ({
                        action: form.action || '',
                        method: form.method || 'get',
                        id: form.id || '',
                        class: form.className || '',
                        inputs: Array.from(form.querySelectorAll('input, select, textarea')).map(input => ({
                            type: input.type || 'text',
                            name: input.name || '',
                            id: input.id || '',
                            required: input.hasAttribute('required'),
                            placeholder: input.placeholder || ''
                        }))
                    }));
                }
            """
            )
        except Exception as e:
            logger.error(f"Error extracting forms: {e}")

        return forms

    async def _extract_media(self, page: Page) -> List[Dict]:
        """
        Extract media elements information
        """
        media = []

        try:
            # Images
            images = await page.evaluate(
                """
                () => {
                    return Array.from(document.querySelectorAll('img')).map(img => ({
                        type: 'image',
                        src: img.src,
                        alt: img.alt || '',
                        title: img.title || '',
                        width: img.naturalWidth,
                        height: img.naturalHeight
                    }));
                }
            """
            )
            media.extend(images)

            # Videos
            videos = await page.evaluate(
                """
                () => {
                    return Array.from(document.querySelectorAll('video')).map(video => ({
                        type: 'video',
                        src: video.src || (video.querySelector('source') ? video.querySelector('source').src : ''),
                        poster: video.poster || ''
                    }));
                }
            """
            )
            media.extend(videos)

        except Exception as e:
            logger.error(f"Error extracting media: {e}")

        return media

    async def _extract_navigation_links(self, page) -> Set[str]:
        """
        Extract links from navigation elements
        """
        links = set()
        try:
            # Find navigation elements
            nav_selectors = [
                "nav",
                '[role="navigation"]',
                ".nav",
                ".navigation",
                "#nav",
                "#navigation",
            ]

            for selector in nav_selectors:
                elements = await page.query_selector_all(f"{selector} a")
                for element in elements:
                    href = await element.get_attribute("href")
                    if href:
                        absolute_url = urljoin(self.base_url, href)
                        if self._is_same_domain(absolute_url):
                            links.add(absolute_url)

            # Also get header and footer links
            for area in ["header", "footer"]:
                elements = await page.query_selector_all(f"{area} a")
                for element in elements:
                    href = await element.get_attribute("href")
                    if href:
                        absolute_url = urljoin(self.base_url, href)
                        if self._is_same_domain(absolute_url):
                            links.add(absolute_url)

        except Exception as e:
            logger.error(f"Error extracting navigation links: {e}")

        return links

    async def _extract_javascript_links(self, page) -> Set[str]:
        """
        Extract links that might be generated by JavaScript
        """
        links = set()
        try:
            # Get all links after JavaScript execution
            all_links = await page.evaluate(
                """
                () => {
                    const links = new Set();
                    document.querySelectorAll('a[href]').forEach(a => {
                        links.add(a.href);
                    });
                    return Array.from(links);
                }
            """
            )

            for link in all_links:
                if self._is_same_domain(link):
                    links.add(link)

        except Exception as e:
            logger.error(f"Error extracting JavaScript links: {e}")

        return links

    async def _scroll_page(self, page):
        """
        Scroll through the page to trigger lazy loading
        """
        try:
            await page.evaluate(
                """
                async () => {
                    const distance = 1000;
                    const delay = 100;
                    const maxScroll = document.body.scrollHeight;
                    
                    for (let currentScroll = 0; currentScroll < maxScroll; currentScroll += distance) {
                        window.scrollTo(0, currentScroll);
                        await new Promise(resolve => setTimeout(resolve, delay));
                    }
                    
                    // Scroll back to top
                    window.scrollTo(0, 0);
                }
            """
            )
        except Exception as e:
            logger.error(f"Error scrolling page: {e}")

    def _generate_common_urls(self) -> Set[str]:
        """
        Generate common URL patterns that many websites use
        """
        common_paths = [
            "/",
            "/about",
            "/about-us",
            "/contact",
            "/contact-us",
            "/products",
            "/services",
            "/pricing",
            "/blog",
            "/news",
            "/faq",
            "/help",
            "/support",
            "/terms",
            "/privacy",
            "/policy",
            "/team",
            "/careers",
            "/jobs",
            "/portfolio",
            "/gallery",
            "/testimonials",
            "/reviews",
            "/api",
            "/docs",
            "/documentation",
            "/resources",
            "/downloads",
            "/features",
            "/solutions",
            "/customers",
            "/partners",
            "/login",
            "/signup",
            "/register",
            "/search",
            "/sitemap",
            "/site-map",
        ]

        generated_urls = set()
        for path in common_paths:
            generated_urls.add(urljoin(self.base_url, path))

        # Also try with .html extensions
        for path in common_paths:
            if not path.endswith("/"):
                generated_urls.add(urljoin(self.base_url, path + ".html"))
                generated_urls.add(urljoin(self.base_url, path + ".php"))

        return generated_urls

    def _is_same_domain(self, url: str) -> bool:
        """
        Check if URL belongs to the same domain
        """
        try:
            parsed = urlparse(url)
            # Handle both www and non-www versions
            url_domain = parsed.netloc.replace("www.", "")
            self_domain = self.domain.replace("www.", "")
            return url_domain == self_domain
        except:
            return False

    def _is_valid_url(self, url: str) -> bool:
        """
        Check if URL is valid
        """
        try:
            parsed = urlparse(url)
            return bool(parsed.netloc) and bool(parsed.scheme)
        except:
            return False

    def _calculate_url_priority(self, url: str) -> int:
        """
        Calculate priority for URL crawling (lower number = higher priority)
        """
        # Homepage has highest priority
        if url == self.base_url or url == self.base_url + "/":
            return 0

        # Important pages get higher priority
        important_patterns = ["product", "service", "pricing", "about", "contact"]
        for pattern in important_patterns:
            if pattern in url.lower():
                return 1

        # Documentation and help pages
        doc_patterns = ["doc", "help", "faq", "support"]
        for pattern in doc_patterns:
            if pattern in url.lower():
                return 2

        # Blog and news are lower priority
        if "blog" in url.lower() or "news" in url.lower():
            return 3

        # Everything else
        return 5

    def _should_crawl(self, url: str) -> bool:
        """
        Determine if URL should be crawled
        """
        # Skip if already visited
        if url in self.visited_urls:
            return False

        # Must be same domain
        if not self._is_same_domain(url):
            return False

        # Skip certain file types
        skip_extensions = {
            ".pdf",
            ".jpg",
            ".jpeg",
            ".png",
            ".gif",
            ".zip",
            ".mp4",
            ".mp3",
            ".doc",
            ".docx",
            ".xls",
            ".xlsx",
        }
        parsed_url = urlparse(url)
        path = parsed_url.path.lower()

        if any(path.endswith(ext) for ext in skip_extensions):
            return False

        # Skip certain URL patterns
        skip_patterns = ["javascript:", "mailto:", "tel:", "#"]
        if any(pattern in url.lower() for pattern in skip_patterns):
            return False

        return True

    async def _prioritize_urls(self, urls: Set[str]) -> List[Tuple[int, str]]:
        """
        Prioritize URLs for crawling
        """
        prioritized = []
        for url in urls:
            if self._should_crawl(url):
                priority = self._calculate_url_priority(url)
                prioritized.append((priority, url))

        # Sort by priority (lower number = higher priority)
        prioritized.sort(key=lambda x: x[0])

        return prioritized

    def _determine_page_type(self, url: str, page: CrawledPage) -> str:
        """
        Determine the type of page based on URL and content
        """
        url_lower = url.lower()

        # Check URL patterns
        for page_type, patterns in self.url_patterns.items():
            for pattern in patterns:
                if re.search(pattern, url_lower):
                    return page_type

        # Check structured data
        if page.structured_data.get("json_ld"):
            for item in page.structured_data["json_ld"]:
                if isinstance(item, dict):
                    schema_type = item.get("@type", "").lower()
                    if "product" in schema_type:
                        return "product"
                    elif "article" in schema_type or "blogposting" in schema_type:
                        return "blog"
                    elif (
                        "organization" in schema_type or "localbusiness" in schema_type
                    ):
                        return "about"

        # Default
        return "general"

    def _calculate_importance(self, page: CrawledPage) -> float:
        """
        Calculate importance score for a page
        """
        score = 1.0

        # Boost homepage
        if page.url == self.base_url or page.url == self.base_url + "/":
            score *= 3.0

        # Boost based on page type
        if page.page_type in ["product", "pricing", "contact"]:
            score *= 2.0
        elif page.page_type in ["about", "service"]:
            score *= 1.5

        # Boost if has structured data
        if page.structured_data:
            score *= 1.2

        # Boost if has forms (likely interactive)
        if page.forms:
            score *= 1.1

        return min(score, 5.0)  # Cap at 5.0
