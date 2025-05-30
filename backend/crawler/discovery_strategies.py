import asyncio
from typing import Set, List
import aiohttp
from urllib.parse import urljoin, urlparse
import xml.etree.ElementTree as ET
import logging
import re

logger = logging.getLogger(__name__)


class DiscoveryStrategies:
    def __init__(self, domain: str):
        self.domain = domain
        self.base_url = f"https://{domain}"

    async def discover_from_sitemap(self) -> Set[str]:
        """Discover URLs from sitemap.xml"""
        urls = set()
        sitemap_urls = [
            f"{self.base_url}/sitemap.xml",
            f"{self.base_url}/sitemap_index.xml",
            f"{self.base_url}/sitemap.xml.gz",
            f"{self.base_url}/sitemap_index.xml.gz",
        ]

        async with aiohttp.ClientSession() as session:
            for sitemap_url in sitemap_urls:
                try:
                    async with session.get(sitemap_url, timeout=10) as response:
                        if response.status == 200:
                            content = await response.text()
                            discovered = self._parse_sitemap(content)
                            urls.update(discovered)

                            # If it's a sitemap index, fetch child sitemaps
                            if "sitemapindex" in content:
                                child_sitemaps = self._extract_sitemap_urls(content)
                                for child_url in child_sitemaps:
                                    try:
                                        async with session.get(
                                            child_url, timeout=10
                                        ) as child_response:
                                            if child_response.status == 200:
                                                child_content = (
                                                    await child_response.text()
                                                )
                                                urls.update(
                                                    self._parse_sitemap(child_content)
                                                )
                                    except:
                                        pass
                            break
                except Exception as e:
                    logger.debug(f"Sitemap not found at {sitemap_url}: {e}")

        return urls

    async def discover_from_robots(self) -> Set[str]:
        """Discover URLs from robots.txt"""
        urls = set()
        robots_url = f"{self.base_url}/robots.txt"

        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(robots_url, timeout=10) as response:
                    if response.status == 200:
                        content = await response.text()
                        for line in content.split("\n"):
                            line = line.strip()
                            if line.startswith("Sitemap:"):
                                sitemap_url = line.split(":", 1)[1].strip()
                                urls.add(sitemap_url)
                            elif line.startswith("Allow:") or line.startswith(
                                "Disallow:"
                            ):
                                path = line.split(":", 1)[1].strip()
                                if path and path != "/" and not path.startswith("*"):
                                    full_url = urljoin(self.base_url, path)
                                    urls.add(full_url)
            except Exception as e:
                logger.debug(f"Robots.txt not found: {e}")

        return urls

    async def discover_via_search(self, page) -> Set[str]:
        """Discover URLs by using site search functionality"""
        urls = set()

        try:
            # Look for search form
            search_form = await page.query_selector(
                'form[role="search"], form.search, form#search'
            )
            if search_form:
                # Try common search terms
                search_terms = ["products", "services", "about", "contact", "help"]

                for term in search_terms[:2]:  # Limit to avoid overwhelming the site
                    try:
                        # Find search input
                        search_input = await page.query_selector(
                            'input[type="search"], input[name="q"], input[name="query"], input[name="search"]'
                        )
                        if search_input:
                            await search_input.fill(term)
                            await search_input.press("Enter")
                            await page.wait_for_load_state("networkidle", timeout=5000)

                            # Extract links from results
                            links = await page.evaluate(
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
                            urls.update(links)

                            # Go back to original page
                            await page.go_back()
                    except:
                        pass

        except Exception as e:
            logger.debug(f"Search discovery failed: {e}")

        return urls

    async def discover_via_search_engines(self) -> Set[str]:
        """Discover URLs via search engines (simplified for now)"""
        # In production, you'd use a search API
        # For now, return empty set to avoid external dependencies
        return set()

    def _parse_sitemap(self, content: str) -> Set[str]:
        """Parse sitemap XML content"""
        urls = set()
        try:
            # Remove namespaces for easier parsing
            content = re.sub(r"xmlns[^>]+", "", content)
            root = ET.fromstring(content)

            # Find all loc elements
            for elem in root.iter():
                if elem.tag.endswith("loc"):
                    url = elem.text
                    if url:
                        urls.add(url.strip())
        except Exception as e:
            logger.error(f"Failed to parse sitemap: {e}")
        return urls

    def _extract_sitemap_urls(self, content: str) -> List[str]:
        """Extract sitemap URLs from sitemap index"""
        urls = []
        try:
            content = re.sub(r"xmlns[^>]+", "", content)
            root = ET.fromstring(content)

            for elem in root.iter():
                if elem.tag.endswith("loc") and elem.text:
                    urls.append(elem.text.strip())
        except:
            pass
        return urls
