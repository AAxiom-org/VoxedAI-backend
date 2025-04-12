"""
Flow module for research digest generation.
"""
import re
import yaml
import json
import asyncio
from typing import List, Dict, Any, Optional, Tuple

from app.core.logging import logger
from app.services.llm_service import llm_service
from app.agents.research.nodes import DigestNode


async def generate_search_queries(notes_content: str, space_id: str, note_ids: List[str]) -> List[Dict[str, Any]]:
    """
    Generate search queries based on the content of notes, grouped by topic.
    
    Args:
        notes_content: Combined content from all notes
        space_id: ID of the space
        note_ids: List of note IDs that contributed to the content
        
    Returns:
        List[Dict[str, Any]]: List of topic groups, each containing a topic and related queries
    """
    try:
        prompt = f"""
        # Search Query Generation Task
        
        You are a research assistant analyzing a collection of notes to generate relevant search queries.
        Your task is to:
        
        1. Analyze the notes content to identify 2-4 distinct key topics or themes
        2. For each topic, generate 2-3 specific, well-formed search queries
        3. Create diverse queries that cover different aspects of each topic
        
        ## Notes Content
        
        {notes_content[:50000]}  # Truncate if too long
        
        ## Output Format
        
        You Must Respond in YAML format:
        ```yaml
        thinking: |
            <your analysis of the main topics and knowledge gaps in the notes>
        action: none
        parameters:
            topic_groups:
                - topic: <topic 1 name>
                  description: <brief description of this topic>
                  related_note_ids: <list of note IDs that are relevant to this topic>
                  queries:
                    - <search query 1.1>
                    - <search query 1.2>
                    - <search query 1.3>
                - topic: <topic 2 name>
                  description: <brief description of this topic>
                  related_note_ids: <list of note IDs that are relevant to this topic>
                  queries:
                    - <search query 2.1>
                    - <search query 2.2>
                - topic: <topic 3 name>
                  description: <brief description of this topic>
                  related_note_ids: <list of note IDs that are relevant to this topic>
                  queries:
                    - <search query 3.1>
                    - <search query 3.2>
                    - <search query 3.3>
        ```
        
        The note IDs to consider are: {note_ids}
        For each topic, analyze which note IDs from the list are most relevant and include them in the related_note_ids field.
        Generate specific, detailed queries that will return valuable information to supplement the existing notes.
        """
        
        # Call the LLM to generate search queries
        response_text = await llm_service._call_llm(
            prompt=prompt,
            model_name="google/gemini-2.0-flash-001",
            stream=False,
            temperature=0.3,
            max_tokens=4000
        )
        
        # Extract YAML from response
        yaml_str = extract_yaml_from_text(response_text)
        result = yaml.safe_load(yaml_str)
        
        topic_groups = result.get("parameters", {}).get("topic_groups", [])
        
        # Ensure we have at least one topic group
        if not topic_groups:
            logger.warning("No topic groups generated, using default topic")
            topic_groups = [{
                "topic": f"Research on {space_id}",
                "description": "General research on the space topics",
                "related_note_ids": note_ids,
                "queries": ["latest research and developments in " + space_id]
            }]
        
        logger.info(f"Generated {len(topic_groups)} topic groups with queries")
        return topic_groups
    
    except Exception as e:
        logger.error(f"Error generating search queries: {e}")
        # Return a default topic group if generation fails
        return [{
            "topic": "General Research",
            "description": "General research on technology and developments",
            "related_note_ids": note_ids,
            "queries": ["latest developments and research in technology"]
        }]


async def perform_web_search(query: str) -> str:
    """
    Perform a web search using GPT-4o with search capability.
    
    Args:
        query: The search query
        
    Returns:
        str: Search results with citations
    """
    try:
        prompt = f"""
        Please search the web for information about:
        
        {query}
        
        Return a comprehensive summary of the information you find, including links to sources.
        Focus on providing factual, up-to-date information from reliable sources.
        Include direct citations to sources as markdown links.
        """
        
        # Call LLM with search capability
        response_text = await llm_service._call_llm(
            prompt=prompt,
            model_name="openai/gpt-4o-mini-search-preview",  # Model with search capability
            stream=False,
            temperature=0.2,
            max_tokens=4000
        )
        
        logger.info(f"Got search results for query: {query}")
        return response_text
    
    except Exception as e:
        logger.error(f"Error performing web search: {e}")
        return f"Error searching for {query}: {str(e)}"


async def generate_digest(notes_content: str, search_results: List[Tuple[str, str]], topic: str, description: str, related_note_ids: List[str]) -> DigestNode:
    """
    Generate a research digest from notes and search results.
    
    Args:
        notes_content: Combined content from all notes
        search_results: List of (query, result) tuples from web searches
        topic: The main topic for this digest
        description: Brief description of the topic
        related_note_ids: List of note IDs related to this topic
        
    Returns:
        DigestNode: Generated digest
    """
    try:
        # Format search results for the prompt
        search_context = ""
        for i, (query, result) in enumerate(search_results):
            search_context += f"\n\n## SEARCH {i+1}: {query}\n\n{result}"
        
        prompt = f"""
        # Research Digest Generation Task
        
        You are a knowledge synthesizer tasked with creating a focused research digest about: {topic}
        Topic description: {description}
        
        Your task is to:
        
        1. Analyze the provided notes and search results related to this specific topic: {topic}
        2. Synthesize a coherent, well-structured digest that is strictly focused on this topic
        3. Include relevant citations and links from the search results
        4. Structure the digest with clear headings, bullet points, and formatting
        
        ## Notes Content
        
        {notes_content[:30000]}  # Truncate if too long
        
        ## Search Results
        {search_context}
        
        ## Output Format
        
        You Must Respond in YAML format:
        ```yaml
        thinking: |
            <your analysis and synthesis process>
        action: none
        parameters:
            title: <a concise, descriptive title for the digest related to "{topic}">
            content: |
                <full markdown content including headings, paragraphs, bullet points, and inline citations>
            links:
                - <link1>
                - <link2>
                # All links that were included in the content, extracted for easy reference
        ```
        
        Create a high-quality, informative digest that is focused on the topic "{topic}".
        Use proper markdown formatting with headings, lists, and emphasis where appropriate.
        """
        
        # Call the LLM to generate the digest
        response_text = await llm_service._call_llm(
            prompt=prompt,
            model_name="google/gemini-2.0-flash-001",
            stream=False,
            temperature=0.4,
            max_tokens=8000
        )
        
        # Extract YAML from response
        yaml_str = extract_yaml_from_text(response_text)
        
        try:
            # Try to parse the YAML
            result = yaml.safe_load(yaml_str)
            
            # Extract digest components
            parameters = result.get("parameters", {})
            title = parameters.get("title", f"Research Digest: {topic}")
            content = parameters.get("content", "No content generated")
            links = parameters.get("links", [])
            
            # If links weren't properly extracted, try to extract them from the content
            if not links:
                links = extract_links_from_markdown(content)
            
        except Exception as yaml_error:
            logger.error(f"Error parsing YAML: {yaml_error}")
            
            # Attempt to extract title, content, and links directly from response text
            title = extract_title_from_text(response_text) or f"Research Digest: {topic}"
            content = extract_content_from_text(response_text)
            links = extract_links_from_markdown(content)
            
            logger.info(f"Extracted content directly: title='{title}', content_length={len(content)}, links={len(links)}")
        
        # Create and return the digest node
        digest = DigestNode(
            title=title,
            content=content,
            all_links=links,
            related_note_ids=related_note_ids
        )
        
        logger.info(f"Generated digest: {title} with {len(links)} links and {len(related_note_ids)} related notes")
        return digest
    
    except Exception as e:
        logger.error(f"Error generating digest: {e}")
        # Return a basic digest if generation fails
        return DigestNode(
            title=f"Error Generating Research Digest: {topic}",
            content=f"An error occurred while generating the digest: {str(e)}",
            all_links=[],
            related_note_ids=related_note_ids
        )


def extract_yaml_from_text(text: str) -> str:
    """
    Extract YAML content from text.
    
    Args:
        text: Text containing YAML
        
    Returns:
        str: Extracted YAML
    """
    # Look for YAML block
    yaml_start = text.find("```yaml")
    if yaml_start != -1:
        yaml_start += 7  # Skip the ```yaml marker
        yaml_end = text.find("```", yaml_start)
        if yaml_end != -1:
            # Extract the YAML content
            yaml_content = text[yaml_start:yaml_end].strip()
            
            # Ensure proper formatting for the thinking section
            if "thinking:" in yaml_content and "|" in yaml_content:
                # Split the content by lines
                lines = yaml_content.split('\n')
                formatted_lines = []
                in_thinking = False
                
                for line in lines:
                    if line.strip().startswith("thinking:"):
                        in_thinking = True
                        formatted_lines.append(line)
                    elif in_thinking and not line.strip().startswith("action:"):
                        # Ensure proper indentation for multiline thinking content
                        if not line.startswith("    ") and line.strip():
                            formatted_lines.append("    " + line)
                        else:
                            formatted_lines.append(line)
                    else:
                        in_thinking = False
                        formatted_lines.append(line)
                
                yaml_content = "\n".join(formatted_lines)
            
            return yaml_content
    
    # If no YAML block found, try to format it as YAML if it contains action:
    if "action:" in text and "thinking:" in text:
        lines = text.split('\n')
        formatted_lines = []
        in_thinking = False
        
        for line in lines:
            if line.strip().startswith("thinking:"):
                in_thinking = True
                formatted_lines.append(line)
            elif in_thinking and not line.strip().startswith("action:"):
                # Ensure proper indentation for multiline thinking content
                if not line.startswith("    ") and line.strip():
                    formatted_lines.append("    " + line)
                else:
                    formatted_lines.append(line)
            else:
                in_thinking = False
                formatted_lines.append(line)
        
        return "\n".join(formatted_lines)
    
    # If no YAML block found and can't format as YAML, return the whole text
    return text


def extract_links_from_markdown(markdown_text: str) -> List[str]:
    """
    Extract all links from markdown content.
    
    Args:
        markdown_text: Markdown content
        
    Returns:
        List[str]: Extracted links
    """
    # Find all markdown links [text](url)
    md_links = re.findall(r'\[.+?\]\((https?://[^\s\)]+)\)', markdown_text)
    
    # Find all direct URLs
    direct_links = re.findall(r'(?<!\()(https?://[^\s\)]+)(?!\))', markdown_text)
    
    # Combine and remove duplicates
    all_links = list(set(md_links + direct_links))
    
    return all_links


def extract_title_from_text(text: str) -> str:
    """
    Extract a title from text when YAML parsing fails.
    
    Args:
        text: The response text
        
    Returns:
        str: Extracted title
    """
    # Look for title in parameters section
    title_match = re.search(r'title:\s*([^\n]+)', text)
    if title_match:
        return title_match.group(1).strip()
    
    # Look for markdown heading
    heading_match = re.search(r'#\s+([^\n]+)', text)
    if heading_match:
        return heading_match.group(1).strip()
    
    # Default title
    return "Research Digest"


def extract_content_from_text(text: str) -> str:
    """
    Extract content from text when YAML parsing fails.
    
    Args:
        text: The response text
        
    Returns:
        str: Extracted content
    """
    # Try to find content section in YAML
    content_match = re.search(r'content:\s*\|\s*\n([\s\S]+?)(?:\n\s*links:|$)', text)
    if content_match:
        content = content_match.group(1)
        # Fix indentation
        lines = content.split('\n')
        # Remove common leading spaces
        if lines and lines[0].startswith(' '):
            spaces = len(lines[0]) - len(lines[0].lstrip())
            content = '\n'.join([line[spaces:] if line.startswith(' ' * spaces) else line for line in lines])
        return content.strip()
    
    # If content section not found, look for markdown headings and extract everything after them
    heading_match = re.search(r'(#\s+[^\n]+[\s\S]+)', text)
    if heading_match:
        return heading_match.group(1).strip()
    
    # If nothing is found, return the original text
    return text
