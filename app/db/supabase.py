"""
Supabase client module for connecting to Supabase.
"""
from typing import Any, Dict, Optional, List
import json

from supabase import Client, create_client

from app.core.config import settings
from app.core.logging import logger


class SupabaseClient:
    """
    Singleton client for Supabase.
    """

    _instance: Optional["SupabaseClient"] = None
    _client: Optional[Client] = None

    def __new__(cls) -> "SupabaseClient":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if self._client is None:
            try:
                self._client = create_client(
                    settings.SUPABASE_URL, settings.SUPABASE_SERVICE_ROLE_KEY
                )
                logger.info("Supabase client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Supabase client: {e}")
                raise

    @property
    def client(self) -> Client:
        """
        Returns the Supabase client.
        
        Returns:
            Client: The Supabase client.
        """
        if self._client is None:
            raise ValueError("Supabase client not initialized")
        return self._client

    async def get_file_metadata(self, file_id: str) -> Dict[str, Any]:
        """
        Fetches file metadata from the file_metadata table.
        
        Args:
            file_id: The ID of the file.
            
        Returns:
            Dict[str, Any]: The file metadata.
        """
        try:
            response = self.client.table("file_metadata").select("*").eq("file_id", file_id).execute()
            if response.data:
                return response.data[0]
            return {}
        except Exception as e:
            logger.error(f"Error fetching file metadata: {e}")
            raise

    async def get_notebook_file(self, file_id: str) -> Dict[str, Any]:
        """
        Fetches file information from the notebook_files table.
        
        Args:
            file_id: The ID of the file.
            
        Returns:
            Dict[str, Any]: The file information.
        """
        try:
            response = self.client.table("notebook_files").select("*").eq("id", file_id).execute()
            if response.data:
                return response.data[0]
            return {}
        except Exception as e:
            logger.error(f"Error fetching notebook file: {e}")
            raise

    async def create_file_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Creates a new file metadata record.
        
        Args:
            metadata: The metadata to create.
            
        Returns:
            Dict[str, Any]: The created metadata.
        """
        try:
            response = self.client.table("file_metadata").insert(metadata).execute()
            if response.data:
                return response.data[0]
            return {}
        except Exception as e:
            logger.error(f"Error creating file metadata: {e}")
            raise

    async def update_file_metadata(self, id: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Updates an existing file metadata record.
        
        Args:
            id: The ID of the metadata record.
            metadata: The metadata to update.
            
        Returns:
            Dict[str, Any]: The updated metadata.
        """
        try:
            response = self.client.table("file_metadata").update(metadata).eq("id", id).execute()
            if response.data:
                return response.data[0]
            return {}
        except Exception as e:
            logger.error(f"Error updating file metadata: {e}")
            raise

    async def fetch_file_from_storage(self, file_path: str) -> bytes:
        """
        Fetches a file from Supabase storage.
        
        Args:
            file_path: The path of the file in storage.
            
        Returns:
            bytes: The file contents.
        """
        try:
            response = self.client.storage.from_("Vox").download(file_path)
            return response
        except Exception as e:
            logger.error(f"Error fetching file from storage: {e}")
            raise

    async def get_user_toggled_files(self, user_id: str) -> List[str]:
        """
        Fetches the toggled files for a user from the users table.
        
        Args:
            user_id: The ID of the user.
            
        Returns:
            Dict[bool, List[str]]: Dictionary with boolean indicating if the file is in file_metadata table (ie exists in pinecone). List of pinecone_ids if true, file_id's if false.
        """
        try:
            toggled_files: Dict[bool, List[str]] = {True: [], False: []}
            response = self.client.table("users").select("toggled_files").eq("id", user_id).execute()
            
            if response.data and response.data[0].get("toggled_files"):
                # toggled_files is stored as JSONB in the database
                file_ids = response.data[0]["toggled_files"]
                
                for file_id in file_ids:
                    file_response = self.client.table("file_metadata").select("pinecone_id").eq("file_id", file_id).execute()
                    
                    if file_response.data and len(file_response.data) > 0 and file_response.data[0].get("pinecone_id"):
                        pinecone_id = file_response.data[0]["pinecone_id"]
                        if pinecone_id:  # Only add non-empty pinecone_ids
                            toggled_files[True].append(pinecone_id)
                            logger.info(f"Added toggled file pinecone_id: {pinecone_id} to toggled_files[True]")
                    else:
                        toggled_files[False].append(file_id)
                        logger.info(f"Added toggled filefile_id: {file_id} to toggled_files[False]")

            return toggled_files
        except Exception as e:
            logger.error(f"Error fetching user toggled files: {e}")
            raise

    async def get_file_content(self, file_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Fetches the content of the files from the Vox bucket and formats it to match
        the structure returned by the retrieve_context function.
        
        Args:
            file_ids: A list of file IDs to fetch.
            
        Returns:
            List[Dict[str, Any]]: A list of context entries with the structure:
            {
                "text": The processed file content,
                "score": A fixed score (50.0),
                "file_id": The ID of the file,
                "file_path": The path of the file,
                "source": "Users Notes: {file_name}",
                "metadata": An empty dictionary
            }
        """
        try:
            context_entries = []
            for file_id in file_ids:
                # Get file details from notebook_files table
                file_details = await self.get_notebook_file(file_id)
                if not file_details:
                    logger.warning(f"File details not found for file_id: {file_id}")
                    continue
                
                file_path = file_details.get("file_path", "")
                file_name = file_details.get("file_name", "Unknown")
                
                # Download file content from storage
                try:
                    response = self.client.storage.from_("Vox").download(file_path)
                    if response:
                        try:
                            json_data = json.loads(response.decode('utf-8'))
                            processed_content = self.process_json(json_data)
                            
                            # Create context entry in the expected format
                            context_entry = {
                                "text": processed_content,
                                "score": 50.0,  # Fixed score as instructed
                                "file_id": file_id,
                                "file_path": file_path,
                                "source": f"Users Notes: {file_name}",
                                "metadata": {}  # Empty metadata as instructed
                            }
                            context_entries.append(context_entry)
                            logger.info(f"DEBUG - Processed file source: {context_entry['source']}")
                            logger.info(f"Successfully processed file: {file_name}")
                        except json.JSONDecodeError as e:
                            logger.error(f"Error parsing file content as JSON: {e}")
                            # Handle non-JSON files
                            text_content = response.decode('utf-8', errors='replace')
                            context_entry = {
                                "text": text_content,
                                "score": 50.0,
                                "file_id": file_id,
                                "file_path": file_path,
                                "source": f"Users Notes: {file_name}",
                                "metadata": {}
                            }
                            context_entries.append(context_entry)
                    else:
                        logger.warning(f"No content found for file: {file_path}")
                except Exception as e:
                    logger.error(f"Error downloading file {file_path}: {e}")
            
            return context_entries
        except Exception as e:
            logger.error(f"Error in get_file_content: {e}")
            raise
    
    def get_file_path(self, file_id: str) -> str:
        """
        Fetches the path of the file from the notebook_files table.
        """
        try:
            response = self.client.table("notebook_files").select("file_path").eq("id", file_id).execute()
            return response.data[0]["file_path"]
        except Exception as e:
            logger.error(f"Error fetching file path: {e}")
            raise

    def extract_text(self, content_list):
        """Extract raw text from a list of content items."""
        texts = []
        for item in content_list:
            # Each item is expected to be a dict with a 'text' key.
            if isinstance(item, dict) and 'text' in item:
                texts.append(item['text'])
        return ' '.join(texts).strip()

    def process_block(self, block):
        """Process a single block and return a standardized string representation."""
        # Handle case where block is a string
        if isinstance(block, str):
            return f"Text: {block}"
            
        block_type = block.get('type')
        if block_type == 'heading':
            text = self.extract_text(block.get('content', []))
            return f"Heading: {text}"
        elif block_type == 'codeBlock':
            language = block.get('props', {}).get('language', 'text')
            text = self.extract_text(block.get('content', []))
            return f"CodeBlock ({language}): {text}"
        elif block_type == 'paragraph':
            text = self.extract_text(block.get('content', []))
            return f"Paragraph: {text if text else '(empty)'}"
        elif block_type == 'table':
            # Tables have a nested structure with rows and cells.
            table_content = block.get('content', {})
            rows = table_content.get('rows', [])
            result_lines = ["Table:"]
            for row in rows:
                cell_texts = []
                cells = row.get('cells', [])
                for cell in cells:
                    # Each cell is a list of content items.
                    cell_text = self.extract_text(cell)
                    cell_texts.append(cell_text if cell_text else "(empty)")
                result_lines.append("  - " + " | ".join(cell_texts))
            return "\n".join(result_lines)
        else:
            # For any other block type, return its type and content if any.
            text = self.extract_text(block.get('content', []))
            return f"{block_type.capitalize()}: {text if text else '(empty)'}"

    def process_section(self, section):
        """Process a section (a list of blocks) and return a list of string representations."""
        # Handle case where section is not a list
        if not isinstance(section, list):
            if isinstance(section, dict):
                return [self.process_block(section)]
            elif isinstance(section, str):
                return [f"Text: {section}"]
            else:
                return [f"Unknown content type: {type(section)}"]
        return [self.process_block(block) for block in section]

    def process_json(self, data):
        """
        Process the JSON data, which is expected to be a list of sections,
        where each section is a list of blocks.
        """
        output_lines = []
        
        # Handle case where data is not a list
        if not isinstance(data, list):
            if isinstance(data, dict):
                output_lines.append("Section 1:")
                for line in self.process_section([data]):
                    output_lines.append("• " + line)
            elif isinstance(data, str):
                output_lines.append("Section 1:")
                output_lines.append(f"• Text: {data}")
            else:
                output_lines.append(f"Unknown data type: {type(data)}")
            return "\n".join(output_lines).strip()
            
        for i, section in enumerate(data):
            output_lines.append(f"Section {i+1}:")
            for line in self.process_section(section):
                output_lines.append("• " + line)
            output_lines.append("")  # Add an empty line between sections
        return "\n".join(output_lines).strip()



# Global instance of the Supabase client
supabase_client = SupabaseClient() 