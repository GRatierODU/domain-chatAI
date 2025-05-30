from typing import Dict, List, Tuple, Optional
from PIL import Image
import numpy as np
from bs4 import BeautifulSoup
import io
import logging

logger = logging.getLogger(__name__)

class LayoutAnalyzer:
    """
    Analyzes page layout and structure
    """
    
    def __init__(self):
        # Layout patterns for different page types
        self.layout_patterns = {
            'header': ['header', 'nav', 'navigation', 'menu', 'top-bar'],
            'footer': ['footer', 'bottom', 'copyright', 'links'],
            'sidebar': ['sidebar', 'aside', 'side-menu', 'widget'],
            'main': ['main', 'content', 'article', 'primary'],
            'hero': ['hero', 'banner', 'jumbotron', 'showcase'],
            'features': ['features', 'benefits', 'services'],
            'testimonials': ['testimonials', 'reviews', 'feedback'],
            'cta': ['cta', 'call-to-action', 'signup', 'subscribe'],
            'pricing': ['pricing', 'plans', 'packages', 'tiers']
        }
        
    async def analyze(self, screenshot: bytes, html: str) -> Dict:
        """
        Analyze page layout from screenshot and HTML
        """
        soup = BeautifulSoup(html, 'html.parser')
        
        # Analyze HTML structure
        structure = self._analyze_html_structure(soup)
        
        # Identify sections
        sections = self._identify_sections(soup)
        
        # Analyze visual hierarchy
        hierarchy = self._analyze_visual_hierarchy(soup)
        
        # Detect layout type
        layout_type = self._detect_layout_type(sections)
        
        return {
            'structure': structure,
            'sections': sections,
            'hierarchy': hierarchy,
            'layout_type': layout_type,
            'navigation': self._extract_navigation(soup),
            'main_content': self._identify_main_content(soup)
        }
        
    def _analyze_html_structure(self, soup: BeautifulSoup) -> Dict:
        """
        Analyze the overall HTML structure
        """
        structure = {
            'depth': self._calculate_max_depth(soup.body) if soup.body else 0,
            'total_elements': len(soup.find_all()),
            'text_elements': len(soup.find_all(text=True)),
            'interactive_elements': len(soup.find_all(['a', 'button', 'input', 'select', 'textarea'])),
            'media_elements': len(soup.find_all(['img', 'video', 'audio', 'svg'])),
            'semantic_elements': len(soup.find_all(['header', 'nav', 'main', 'article', 'section', 'aside', 'footer']))
        }
        
        return structure
        
    def _identify_sections(self, soup: BeautifulSoup) -> List[Dict]:
        """
        Identify major sections of the page
        """
        sections = []
        
        # Look for semantic HTML5 elements first
        for tag in ['header', 'nav', 'main', 'article', 'section', 'aside', 'footer']:
            elements = soup.find_all(tag)
            for elem in elements:
                sections.append({
                    'type': tag,
                    'id': elem.get('id', ''),
                    'class': ' '.join(elem.get('class', [])),
                    'text_preview': elem.get_text()[:100].strip(),
                    'importance': self._calculate_importance(elem)
                })
                
        # Look for divs with meaningful classes/ids
        for pattern_type, patterns in self.layout_patterns.items():
            for pattern in patterns:
                # Check classes
                elements = soup.find_all(class_=lambda c: c and pattern in str(c).lower())
                for elem in elements:
                    if not any(s['type'] == pattern_type for s in sections):
                        sections.append({
                            'type': pattern_type,
                            'id': elem.get('id', ''),
                            'class': ' '.join(elem.get('class', [])),
                            'text_preview': elem.get_text()[:100].strip(),
                            'importance': self._calculate_importance(elem)
                        })
                        
                # Check IDs
                elements = soup.find_all(id=lambda i: i and pattern in i.lower())
                for elem in elements:
                    if not any(s['type'] == pattern_type for s in sections):
                        sections.append({
                            'type': pattern_type,
                            'id': elem.get('id', ''),
                            'class': ' '.join(elem.get('class', [])),
                            'text_preview': elem.get_text()[:100].strip(),
                            'importance': self._calculate_importance(elem)
                        })
                        
        return sorted(sections, key=lambda x: x['importance'], reverse=True)
        
    def _analyze_visual_hierarchy(self, soup: BeautifulSoup) -> List[Dict]:
        """
        Analyze visual hierarchy based on heading structure
        """
        hierarchy = []
        
        for level in range(1, 7):  # h1 to h6
            headings = soup.find_all(f'h{level}')
            for heading in headings:
                hierarchy.append({
                    'level': level,
                    'text': heading.get_text().strip(),
                    'parent_section': self._find_parent_section(heading),
                    'has_id': bool(heading.get('id')),
                    'has_anchor': bool(heading.find('a'))
                })
                
        return hierarchy
        
    def _detect_layout_type(self, sections: List[Dict]) -> str:
        """
        Detect the type of layout used
        """
        section_types = [s['type'] for s in sections]
        
        if 'sidebar' in section_types:
            if section_types.count('sidebar') == 2:
                return 'three-column'
            else:
                return 'two-column'
        elif 'hero' in section_types and 'features' in section_types:
            return 'landing-page'
        elif 'article' in section_types:
            return 'article-page'
        elif 'pricing' in section_types:
            return 'pricing-page'
        else:
            return 'single-column'
            
    def _extract_navigation(self, soup: BeautifulSoup) -> Dict:
        """
        Extract navigation structure
        """
        nav_data = {
            'primary': [],
            'secondary': [],
            'breadcrumbs': [],
            'footer_links': []
        }
        
        # Primary navigation
        nav = soup.find('nav') or soup.find(class_='nav') or soup.find(class_='navigation')
        if nav:
            links = nav.find_all('a')
            nav_data['primary'] = [
                {
                    'text': link.get_text().strip(),
                    'href': link.get('href', ''),
                    'is_active': 'active' in link.get('class', [])
                }
                for link in links
            ]
            
        # Breadcrumbs
        breadcrumbs = soup.find(class_='breadcrumb') or soup.find(class_='breadcrumbs')
        if breadcrumbs:
            links = breadcrumbs.find_all('a')
            nav_data['breadcrumbs'] = [
                {
                    'text': link.get_text().strip(),
                    'href': link.get('href', '')
                }
                for link in links
            ]
            
        # Footer links
        footer = soup.find('footer')
        if footer:
            links = footer.find_all('a')
            nav_data['footer_links'] = [
                {
                    'text': link.get_text().strip(),
                    'href': link.get('href', '')
                }
                for link in links[:20]  # Limit to 20 links
            ]
            
        return nav_data
        
    def _identify_main_content(self, soup: BeautifulSoup) -> Dict:
        """
        Identify the main content area
        """
        # Try semantic elements first
        main = soup.find('main') or soup.find('article') or soup.find(role='main')
        
        if not main:
            # Look for common class names
            for class_name in ['main-content', 'main', 'content', 'primary', 'article-body']:
                main = soup.find(class_=class_name)
                if main:
                    break
                    
        if not main:
            # Fallback: find largest text block
            all_divs = soup.find_all('div')
            if all_divs:
                main = max(all_divs, key=lambda div: len(div.get_text()))
                
        if main:
            return {
                'tag': main.name,
                'id': main.get('id', ''),
                'class': ' '.join(main.get('class', [])),
                'text_length': len(main.get_text()),
                'has_headings': bool(main.find_all(['h1', 'h2', 'h3'])),
                'has_images': bool(main.find_all('img')),
                'has_links': bool(main.find_all('a'))
            }
        else:
            return {}
            
    def _calculate_importance(self, element) -> float:
        """
        Calculate importance score for an element
        """
        score = 1.0
        
        # Boost for semantic elements
        if element.name in ['header', 'main', 'article']:
            score *= 2.0
        elif element.name in ['nav', 'section']:
            score *= 1.5
            
        # Boost for having an ID
        if element.get('id'):
            score *= 1.2
            
        # Boost based on content
        text_length = len(element.get_text())
        if text_length > 500:
            score *= 1.5
        elif text_length > 200:
            score *= 1.2
            
        # Boost for having headings
        if element.find_all(['h1', 'h2', 'h3']):
            score *= 1.3
            
        return score
        
    def _calculate_max_depth(self, element, current_depth=0) -> int:
        """
        Calculate maximum depth of HTML tree
        """
        if not element or not hasattr(element, 'children'):
            return current_depth
            
        max_child_depth = current_depth
        for child in element.children:
            if hasattr(child, 'name'):
                child_depth = self._calculate_max_depth(child, current_depth + 1)
                max_child_depth = max(max_child_depth, child_depth)
                
        return max_child_depth
        
    def _find_parent_section(self, element) -> str:
        """
        Find the parent section of an element
        """
        parent = element.parent
        while parent:
            if parent.name in ['section', 'article', 'div']:
                if parent.get('id'):
                    return parent.get('id')
                elif parent.get('class'):
                    return ' '.join(parent.get('class', []))
            parent = parent.parent
        return 'root'