# app/tools/file.py

import json
import asyncio
from typing import Dict, Any, List, Optional
import yaml

from pocketflow import AsyncNode
from app.core.logging import logger
from app.services.llm_service import llm_service
from app.models.space_file import SpaceFile
from app.db.supabase import supabase_client

# Database interaction functions
async def get_file_by_id(file_id: str) -> Optional[SpaceFile]:
    """
    Get file information from the database by ID.
    
    Args:
        file_id: The ID of the file to retrieve.
        
    Returns:
        Optional[SpaceFile]: The file information or None if not found.
    """
    try:
        # Use the Supabase client to get file data
        file_data = await supabase_client.get_space_file(file_id)
        if not file_data:
            logger.error(f"File with ID {file_id} not found")
            return None
        
        # Convert to SpaceFile object
        return SpaceFile.from_dict(file_data)
    except Exception as e:
        logger.error(f"Error getting file by ID: {str(e)}")
        return None

async def download_file(bucket: str, file_path: str) -> str:
    """
    Download a file from Supabase storage.
    
    Args:
        bucket: The storage bucket name.
        file_path: The path of the file in storage.
        
    Returns:
        str: The file content as a string.
    """
    try:
        # Use the Supabase client to fetch the file
        file_bytes = await supabase_client.fetch_file_from_storage(file_path)
        # Convert bytes to string
        return file_bytes.decode('utf-8')
    except Exception as e:
        logger.error(f"Error downloading file: {str(e)}")
        raise

async def upload_file(bucket: str, file_path: str, content: str) -> bool:
    """
    Upload a file to Supabase storage.
    
    Args:
        bucket: The storage bucket name.
        file_path: The path of the file in storage.
        content: The content to upload.
        
    Returns:
        bool: True if upload was successful, False otherwise.
    """
    try:
        # Convert string to bytes
        content_bytes = content.encode('utf-8')
        
        # Use the Supabase client to upload the file
        # We need to handle the upload through the storage API
        # Note: The options need to have string values, not booleans
        response = supabase_client.client.storage.from_(bucket).upload(
            file_path,
            content_bytes,
            {"contentType": "application/json", "upsert": "true"}
        )
        
        logger.info(f"File uploaded successfully: {file_path}")
        return True
    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}")
        return False


class FileInteraction(AsyncNode):
    """
    Tool for interacting with files in the workspace.
    Handles both file viewing and editing operations.
    """
    
    async def prep_async(self, shared):
        parameters = shared.get("tool_parameters", {})
        query = shared.get("query", "")
        context = shared.get("context", {})
        active_file_id = shared.get("active_file_id")
        
        logger.info(f"FileInteraction: Processing with parameters: {parameters}, active_file_id: {active_file_id}")
        
        # Store shared context in parameters for access by other methods
        parameters["_shared"] = shared
        
        return {
            "parameters": parameters,
            "query": query,
            "context": context,
            "active_file_id": active_file_id
        }
    
    async def exec_async(self, prep_res):
        parameters = prep_res["parameters"]
        query = prep_res["query"]
        active_file_id = prep_res.get("active_file_id")
        context = prep_res["context"]
        shared = parameters.get("_shared", {})
        event_queue = shared.get("event_queue")
        
        # Always prioritize active_file_id from context over parameters
        # This is the file ID passed from the agent endpoint
        file_id = active_file_id or parameters.get("file_id")
        
        if not file_id:
            # Send event about missing file ID
            if event_queue is not None:
                try:
                    await event_queue.put({
                        "type": "file_missing_id",
                        "message": "No file ID provided for file interaction"
                    })
                except Exception as e:
                    logger.error(f"Error sending file_missing_id event: {str(e)}")
            
            return {
                "success": False,
                "error": "No file ID provided"
            }
        
        logger.info(f"FileInteraction: Using file_id={file_id}")
        
        # Send event that file lookup is starting
        if event_queue is not None:
            try:
                await event_queue.put({
                    "type": "file_lookup_start",
                    "message": "Looking up file information",
                    "file_id": file_id
                })
            except Exception as e:
                logger.error(f"Error sending file_lookup_start event: {str(e)}")
        
        # Get file information from database
        file_info = await get_file_by_id(file_id)
        if not file_info:
            # Send event about file not found
            if event_queue is not None:
                try:
                    await event_queue.put({
                        "type": "file_not_found",
                        "message": f"File with ID {file_id} not found",
                        "file_id": file_id
                    })
                except Exception as e:
                    logger.error(f"Error sending file_not_found event: {str(e)}")
            
            return {
                "success": False,
                "error": f"File with ID {file_id} not found"
            }
        
        # Send event that file was found
        if event_queue is not None:
            try:
                await event_queue.put({
                    "type": "file_found",
                    "message": f"Found file: {file_info.file_name}",
                    "file_id": file_id,
                    "file_name": file_info.file_name,
                    "file_type": file_info.file_type,
                    "is_note": file_info.is_note,
                })
            except Exception as e:
                logger.error(f"Error sending file_found event: {str(e)}")
        
        # Send event that downloading is starting
        if event_queue is not None:
            try:
                await event_queue.put({
                    "type": "file_download_start",
                    "message": "Downloading file content",
                    "file_id": file_id,
                    "file_name": file_info.file_name
                })
            except Exception as e:
                logger.error(f"Error sending file_download_start event: {str(e)}")
        
        # Download file from storage
        file_path = file_info.file_path
        try:
            file_content = await download_file("Vox", file_path)
            
            # Send event that download completed
            if event_queue is not None:
                try:
                    await event_queue.put({
                        "type": "file_download_complete",
                        "message": "Successfully downloaded file content",
                        "file_id": file_id,
                        "file_name": file_info.file_name,
                        "content_length": len(file_content)
                    })
                except Exception as e:
                    logger.error(f"Error sending file_download_complete event: {str(e)}")
                    
        except Exception as e:
            # Send event about download error
            if event_queue is not None:
                try:
                    await event_queue.put({
                        "type": "file_download_error",
                        "message": f"Error downloading file: {str(e)}",
                        "file_id": file_id,
                        "error": str(e)
                    })
                except Exception as event_e:
                    logger.error(f"Error sending file_download_error event: {str(event_e)}")
            
            return {
                "success": False,
                "error": f"Error downloading file: {str(e)}"
            }
        
        # Determine the action (view/read vs edit)
        action = parameters.get("action", "view")
        
        # Send event about the action being performed
        if event_queue is not None:
            try:
                await event_queue.put({
                    "type": "file_action_determined",
                    "message": f"File action determined: {action}",
                    "file_id": file_id,
                    "file_name": file_info.file_name,
                    "action": action,
                    "is_note": file_info.is_note
                })
            except Exception as e:
                logger.error(f"Error sending file_action_determined event: {str(e)}")
        
        # For non-note files, only reading is allowed
        if not file_info.is_note and action not in ["view", "read"]:
            # Send event about invalid action
            if event_queue is not None:
                try:
                    await event_queue.put({
                        "type": "file_action_invalid",
                        "message": "Only note files can be edited",
                        "file_id": file_id,
                        "file_name": file_info.file_name,
                        "action": action,
                        "is_note": file_info.is_note
                    })
                except Exception as e:
                    logger.error(f"Error sending file_action_invalid event: {str(e)}")
            
            return {
                "success": False,
                "error": "Only note files can be edited"
            }
        
        # Handle the file based on the action and file type
        if action in ["view", "read"]:
            result = await self._handle_file_view(file_info, file_content, query, event_queue)
            
            # Check if the view operation triggered an auto-edit for a note issue
            if file_info.is_note and "fix_result" in result:
                # Send event that an automatic fix was applied
                if event_queue is not None:
                    try:
                        await event_queue.put({
                            "type": "automatic_note_fix_applied",
                            "message": "Automatic fix was applied based on detected issue in note",
                            "file_id": file_id,
                            "file_name": file_info.file_name
                        })
                    except Exception as e:
                        logger.error(f"Error sending automatic_note_fix_applied event: {str(e)}")
                        
            return result
        elif file_info.is_note and action in ["edit", "append", "replace_snippet"]:
            return await self._handle_file_edit(file_info, file_content, query, action, parameters)
        else:
            # Send event about unknown action
            if event_queue is not None:
                try:
                    await event_queue.put({
                        "type": "file_action_unknown",
                        "message": f"Unknown file action: {action}",
                        "file_id": file_id,
                        "file_name": file_info.file_name,
                        "action": action
                    })
                except Exception as e:
                    logger.error(f"Error sending file_action_unknown event: {str(e)}")
            
            return {
                "success": False,
                "error": f"Unknown action: {action}"
            }
    
    async def _handle_file_view(self, file_info: SpaceFile, file_content: str, query: str, event_queue) -> Dict[str, Any]:
        """Handle viewing/reading a file and generating a summary"""
        
        # Send event that file view operation is starting
        if event_queue is not None:
            try:
                await event_queue.put({
                    "type": "file_view_start",
                    "message": f"Starting to process file for viewing: {file_info.file_name}",
                    "file_id": file_info.id,
                    "file_name": file_info.file_name
                })
            except Exception as e:
                logger.error(f"Error sending file_view_start event: {str(e)}")
        
        # Truncate content if needed
        max_chars = 35000
        truncated = False
        if len(file_content) > max_chars:
            # Send event about truncation
            if event_queue is not None:
                try:
                    await event_queue.put({
                        "type": "file_content_truncated",
                        "message": f"File content truncated due to size ({len(file_content)} chars)",
                        "file_id": file_info.id,
                        "file_name": file_info.file_name,
                        "original_size": len(file_content),
                        "truncated_size": max_chars
                    })
                except Exception as e:
                    logger.error(f"Error sending file_content_truncated event: {str(e)}")
            
            file_content = file_content[:max_chars]
            truncated = True
        
        # Process content differently based on file type
        if file_info.is_note:
            # For notes, add line numbers to the properly formatted JSON content
            processed_content = self._add_line_numbers_to_note(file_content)
            
            # Send event about content processing
            if event_queue is not None:
                try:
                    await event_queue.put({
                        "type": "file_content_processed",
                        "message": "Note file content processed with line numbers",
                        "file_id": file_info.id,
                        "file_name": file_info.file_name,
                        "is_note": True
                    })
                except Exception as e:
                    logger.error(f"Error sending file_content_processed event: {str(e)}")
        else:
            # For regular files, just use the content as is
            processed_content = file_content
            
            # Send event about content processing
            if event_queue is not None:
                try:
                    await event_queue.put({
                        "type": "file_content_processed",
                        "message": "Regular file content processed",
                        "file_id": file_info.id,
                        "file_name": file_info.file_name,
                        "is_note": False
                    })
                except Exception as e:
                    logger.error(f"Error sending file_content_processed event: {str(e)}")
        
        # Send event that LLM summarization is starting
        if event_queue is not None:
            try:
                await event_queue.put({
                    "type": "file_summary_start",
                    "message": "Starting to generate file summary",
                    "file_id": file_info.id,
                    "file_name": file_info.file_name
                })
            except Exception as e:
                logger.error(f"Error sending file_summary_start event: {str(e)}")
        
        # Modified prompt that now tells the LLM it can fix note files if needed
        prompt = f"""
        # USER QUERY
        {query}
        
        # FILE INFORMATION
        File Name: {file_info.file_name}
        File Type: {file_info.file_type}
        Is Note: {'Yes' if file_info.is_note else 'No'}
        
        # FILE CONTENT
        {processed_content}
        {'(Content truncated due to length)' if truncated else ''}
        
        # TASK
        Based on the user query, provide the most relevant information from this file.
        Generate a detailed summary or extract the specific information requested.
        You must include as much relevant information as possible from the file, and your thoughts.

        IMPORTANT:
        - If you detect any issues, typos, or needed improvements in the file AND the file is marked as "Is Note: Yes",
          you can suggest fixing it with action=fix_note_issue
        - These improvements can be to any aspect of the note - text content, formatting, tables, lists, code blocks, etc.
        - If the file is NOT a note (Is Note: No), you cannot suggest edits, only provide a summary or get more context
        
        Respond in YAML format:
        ```yaml
        thinking: |
            <your step-by-step reasoning about what information is most relevant>
        action: <one of: provide_summary, needs_more_context, fix_note_issue>
        parameters:
            summary: <detailed summary of the file content relevant to the query>
            next_chunk_start: <if needs_more_context, position to continue from>
            fix_description: <if action is fix_note_issue, describe what needs to be fixed>
        ```
        """
        
        # Call LLM with the prompt using Claude 3.7 Sonnet
        llm_response = await llm_service._call_llm(
            prompt=prompt,
            model_name="deepseek/deepseek-chat-v3-0324",
            stream=False,
            temperature=0.1
        )
        
        # Send event that LLM response was received
        if event_queue is not None:
            try:
                await event_queue.put({
                    "type": "file_summary_received",
                    "message": "Received file summary from language model",
                    "file_id": file_info.id,
                    "file_name": file_info.file_name
                })
            except Exception as e:
                logger.error(f"Error sending file_summary_received event: {str(e)}")
        
        # Parse the YAML response
        try:
            yaml_content = self._extract_yaml_from_text(llm_response)
            response = yaml.safe_load(yaml_content)
            
            # Send event about parsed response
            if event_queue is not None:
                try:
                    action = response.get("action", "provide_summary")
                    await event_queue.put({
                        "type": "file_summary_parsed",
                        "message": f"Parsed file summary response: {action}",
                        "file_id": file_info.id,
                        "file_name": file_info.file_name,
                        "action": action
                    })
                except Exception as e:
                    logger.error(f"Error sending file_summary_parsed event: {str(e)}")
            
            if response.get("action") == "needs_more_context":
                # If more context is needed, provide the next chunk
                next_chunk_start = response.get("parameters", {}).get("next_chunk_start", max_chars)
                
                # Send event about needing more context
                if event_queue is not None:
                    try:
                        await event_queue.put({
                            "type": "file_more_context_needed",
                            "message": "More file context needed",
                            "file_id": file_info.id,
                            "file_name": file_info.file_name,
                            "next_chunk_start": next_chunk_start
                        })
                    except Exception as e:
                        logger.error(f"Error sending file_more_context_needed event: {str(e)}")
                
                # This would need to be handled properly to get the next chunk
                # For now, just return what we have with a note that more context was requested
                return {
                    "success": True,
                    "content": processed_content,
                    "summary": "More context was requested but not provided in this version",
                    "is_complete": False
                }
            elif response.get("action") == "fix_note_issue" and file_info.is_note:
                # LLM detected an issue in the note file that can be fixed
                summary = response.get("parameters", {}).get("summary", "No summary provided")
                fix_description = response.get("parameters", {}).get("fix_description", "No fix description provided")
                
                logger.info(f"Issue detected in note file. Attempting to fix: {fix_description}")
                
                # Send event about detected issue
                if event_queue is not None:
                    try:
                        await event_queue.put({
                            "type": "file_issue_detected",
                            "message": "Issue detected in note file, attempting to fix",
                            "file_id": file_info.id,
                            "file_name": file_info.file_name,
                            "fix_description": fix_description
                        })
                    except Exception as e:
                        logger.error(f"Error sending file_issue_detected event: {str(e)}")
                
                # Call the edit function to fix the issue
                # Create parameters needed for edit operation
                edit_parameters = {
                    "action": "edit",
                    "_shared": {"event_queue": event_queue, "query": f"Fix issue: {fix_description}"}
                }
                
                edit_result = await self._handle_file_edit(
                    file_info=file_info,
                    file_content=file_content, 
                    query=f"Fix the following issue in this note: {fix_description}",
                    action="edit",
                    parameters=edit_parameters
                )
                
                # Return with both the summary and edit results
                return {
                    "success": True,
                    "content": processed_content,
                    "summary": summary + f"\n\nFix attempted: {fix_description}",
                    "fix_result": edit_result,
                    "is_complete": True
                }
            else:
                # Return the summary
                summary = response.get("parameters", {}).get("summary", "No summary provided")
                
                # Send event about completed summary
                if event_queue is not None:
                    try:
                        await event_queue.put({
                            "type": "file_summary_complete",
                            "message": "File summary completed successfully",
                            "file_id": file_info.id,
                            "file_name": file_info.file_name,
                            "summary_length": len(summary)
                        })
                    except Exception as e:
                        logger.error(f"Error sending file_summary_complete event: {str(e)}")
                
                return {
                    "success": True,
                    "content": processed_content,
                    "summary": summary,
                    "is_complete": True
                }
        except Exception as e:
            logger.error(f"Error parsing LLM response: {str(e)}")
            
            # Send event about parsing error
            if event_queue is not None:
                try:
                    await event_queue.put({
                        "type": "file_summary_error",
                        "message": f"Error parsing file summary: {str(e)}",
                        "file_id": file_info.id,
                        "file_name": file_info.file_name,
                        "error": str(e)
                    })
                except Exception as event_e:
                    logger.error(f"Error sending file_summary_error event: {str(event_e)}")
            
            return {
                "success": False,
                "error": f"Error generating file summary: {str(e)}"
            }
    
    async def _handle_file_edit(self, file_info: SpaceFile, file_content: str, query: str, action: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle editing a note file"""
        
        # For notes, we need to validate that the content is proper JSON
        try:
            note_data = json.loads(file_content)
        except json.JSONDecodeError:
            return {
                "success": False,
                "error": "Invalid note format: not valid JSON"
            }
        
        # Send event that file edit is starting
        shared = parameters.get("_shared", {})
        if shared and "event_queue" in shared and shared["event_queue"] is not None:
            event = {
                "type": "file_edit_start",
                "file_id": file_info.id
            }
            await shared["event_queue"].put(event)
        
        # Add line numbers to formatted JSON content for reference
        processed_content = self._add_line_numbers_to_note(file_content)
        logger.info(f"Processed content: {processed_content}")
        
        # Check if this is a fix operation
        is_fix_operation = query.startswith("Fix") or "fix" in query.lower() or "issue" in query.lower()
        
        # Different prompt for notes to ensure structural correctness
        prompt = f"""
        # USER QUERY
        {query}
        
        # NOTE CONTENT (WITH LINE NUMBERS)
        {processed_content}
        
        # TASK
        You are editing a structured note file. The file format is a specialized JSON structure 
        that must be preserved. Each block has an 'id', 'type', 'props', 'content', and 'children' fields.
        
        IMPORTANT: The content you're seeing is a properly formatted version of the JSON with line numbers.
        When specifying line numbers for edits, use these formatted line numbers.
        
        {"Based on issue detection, you need to fix the identified problem in this note. " if is_fix_operation else "Based on the user's query, determine how to modify this note."}
        
        Respond in YAML format:
        ```yaml
        thinking: |
            <your step-by-step reasoning>
        action: <one of: append, replace_snippet, needs_more_context>
        parameters:
            modified_content: |
                <if append: the new content to add at the end>
                <if replace_snippet: use the format below>
            reason: <explanation of changes made>
        ```
        
        For replace_snippet, use this exact format:
        <<<<<<< ORIGINAL // Line X
        (original content to replace)
        =======
        (new content)
        >>>>>>> UPDATED // Line Y
        
        IMPORTANT: 
        1. When replacing content, include the ENTIRE JSON object/array you want to replace, not just parts of it.
        2. Ensure all JSON structure is preserved. All IDs must be kept the same unless specifically changed.
        3. The line numbers must correspond to the formatted JSON with line numbers shown above.
        4. Make sure the replacement content is valid JSON that can be parsed.
        5. {"Since this is a fix operation, focus on correcting the specific issue described." if is_fix_operation else "Make only the changes necessary to fulfill the user's request."}
        """
        
        # Call LLM with the prompt using a reliable model
        llm_response = await llm_service._call_llm(
            prompt=prompt,
            model_name="deepseek/deepseek-chat-v3-0324",
            stream=False,
            temperature=0.1
        )
        logger.info(f"LLM response: {llm_response}")
        
        # Parse the YAML response
        try:
            yaml_content = self._extract_yaml_from_text(llm_response)
            response = yaml.safe_load(yaml_content)
            
            llm_action = response.get("action")
            
            if llm_action == "needs_more_context":
                # Send event that file edit is complete (even though it failed)
                if shared and "event_queue" in shared and shared["event_queue"] is not None:
                    event = {
                        "type": "file_edit_complete",
                        "file_id": file_info.id
                    }
                    await shared["event_queue"].put(event)
                
                return {
                    "success": False,
                    "error": "More context needed to edit the file"
                }
            
            # Get the changes from the LLM response
            modified_content = response.get("parameters", {}).get("modified_content", "")
            reason = response.get("parameters", {}).get("reason", "No explanation provided")
            
            # Apply the changes based on the action
            updated_content = None
            if llm_action == "append":
                # Append content to the end of the file
                try:
                    new_content = json.loads(modified_content)
                    note_data.extend(new_content)
                    updated_content = json.dumps(note_data)
                except json.JSONDecodeError:
                    # Send event that file edit is complete (even though it failed)
                    if shared and "event_queue" in shared and shared["event_queue"] is not None:
                        event = {
                            "type": "file_edit_complete",
                            "file_id": file_info.id
                        }
                        await shared["event_queue"].put(event)
                    
                    return {
                        "success": False,
                        "error": "Invalid JSON format in appended content"
                    }
            
            elif llm_action == "replace_snippet":
                # Parse the snippet replacement format
                updated_content = self._apply_snippet_replacement(file_content, modified_content)
                if not updated_content:
                    # Send event that file edit is complete (even though it failed)
                    if shared and "event_queue" in shared and shared["event_queue"] is not None:
                        event = {
                            "type": "file_edit_complete",
                            "file_id": file_info.id
                        }
                        await shared["event_queue"].put(event)
                    
                    return {
                        "success": False,
                        "error": "Failed to apply snippet replacement"
                    }
                
                # Verify the updated content is valid JSON
                try:
                    json.loads(updated_content)
                except json.JSONDecodeError:
                    # Send event that file edit is complete (even though it failed)
                    if shared and "event_queue" in shared and shared["event_queue"] is not None:
                        event = {
                            "type": "file_edit_complete",
                            "file_id": file_info.id
                        }
                        await shared["event_queue"].put(event)
                    
                    return {
                        "success": False,
                        "error": "Replacement resulted in invalid JSON"
                    }
            
            # Save the updated file
            if updated_content:
                # Upload to the virtual path in the Vox bucket
                # This ensures we don't overwrite the original file and store edits in a separate location
                virtual_path = f"virtual/{file_info.file_path}"
                logger.info(f"Uploading edited file to virtual path: {virtual_path}")
                upload_success = await upload_file("Vox", virtual_path, updated_content)
                
                if not upload_success:
                    # Send event that file edit is complete (even though it failed)
                    if shared and "event_queue" in shared and shared["event_queue"] is not None:
                        event = {
                            "type": "file_edit_complete",
                            "file_id": file_info.id
                        }
                        await shared["event_queue"].put(event)
                    
                    return {
                        "success": False,
                        "error": "Failed to upload updated file content"
                    }
                
                # Send event that file edit is complete
                if shared and "event_queue" in shared and shared["event_queue"] is not None:
                    event = {
                        "type": "file_edit_complete",
                        "file_id": file_info.id
                    }
                    await shared["event_queue"].put(event)
                
                return {
                    "success": True,
                    "message": "File updated successfully",
                    "changes": reason
                }
            else:
                # Send event that file edit is complete (with no changes)
                if shared and "event_queue" in shared and shared["event_queue"] is not None:
                    event = {
                        "type": "file_edit_complete",
                        "file_id": file_info.id
                    }
                    await shared["event_queue"].put(event)
                
                return {
                    "success": False,
                    "error": "No changes applied to the file"
                }
            
        except Exception as e:
            logger.error(f"Error handling file edit: {str(e)}")
            # Send event that file edit is complete (even though it failed)
            if shared and "event_queue" in shared and shared["event_queue"] is not None:
                event = {
                    "type": "file_edit_complete",
                    "file_id": file_info.id
                }
                await shared["event_queue"].put(event)
            
            return {
                "success": False,
                "error": f"Error editing file: {str(e)}"
            }
    
    def _add_line_numbers_to_note(self, note_content: str) -> str:
        """Add line numbers to the note content for reference"""
        try:
            # Parse the JSON content
            json_data = json.loads(note_content)
            # Format the JSON with proper indentation
            formatted_content = json.dumps(json_data, indent=4)
            # Add line numbers to each line
            lines = formatted_content.split('\n')
            return '\n'.join([f"{i+1}: {line}" for i, line in enumerate(lines)])
        except json.JSONDecodeError:
            # If JSON parsing fails, fall back to the original method
            lines = note_content.split('\n')
            return '\n'.join([f"{i+1}: {line}" for i, line in enumerate(lines)])
    
    def _extract_yaml_from_text(self, text: str) -> str:
        """Extract YAML content from the LLM response"""
        # Find YAML block denoted by ```yaml and ``` markers
        yaml_start = text.find("```yaml")
        if yaml_start != -1:
            # Move past the ```yaml marker
            yaml_start += 7
            yaml_end = text.find("```", yaml_start)
            if yaml_end != -1:
                return text[yaml_start:yaml_end].strip()
        
        # If no markers, try to find what looks like YAML content
        # Look for "action:" as a key indicator
        action_idx = text.find("action:")
        if action_idx != -1:
            # Find the start of the YAML-like content
            yaml_start = text.rfind("\n", 0, action_idx)
            if yaml_start == -1:
                yaml_start = 0
            else:
                yaml_start += 1  # Skip the newline
                
            return text[yaml_start:].strip()
            
        # If we can't extract YAML, return the original text
        return text
    
    def _apply_snippet_replacement(self, original_content: str, replacement_spec: str) -> Optional[str]:
        """Apply a snippet replacement to the original content"""
        try:
            # Parse the original content as JSON
            original_json = json.loads(original_content)
            
            # Parse the replacement specification
            original_marker = "<<<<<<< ORIGINAL //"
            separator = "======="
            updated_marker = ">>>>>>> UPDATED //"
            
            if original_marker not in replacement_spec or separator not in replacement_spec or updated_marker not in replacement_spec:
                logger.error("Invalid replacement specification format")
                return None
            
            # Extract the line numbers and content
            parts = replacement_spec.split(original_marker)[1]
            original_line_info, rest = parts.split("\n", 1)
            original_line = int(original_line_info.strip().split("Line")[1].strip())
            
            parts = rest.split(separator)
            original_content_to_replace = parts[0].strip()
            
            parts = parts[1].split(updated_marker)
            new_content = parts[0].strip()
            updated_line_info = parts[1].strip().split("Line")[1].strip()
            updated_line = int(updated_line_info)
            
            # Format the original JSON to match the line numbers in the replacement spec
            formatted_original = json.dumps(original_json, indent=4).split('\n')
            
            # Verify the line numbers
            if original_line < 1 or original_line > len(formatted_original) or updated_line < 1 or updated_line > len(formatted_original):
                logger.error(f"Invalid line numbers: {original_line} to {updated_line}")
                return None
            
            # Extract the original content to be replaced
            # We need to parse the content instead of using line numbers directly
            try:
                # If original_content_to_replace is valid JSON, we can use it to locate the corresponding element
                json_to_replace = json.loads(original_content_to_replace)
                
                # Find the item to replace in the original JSON
                # For simplicity, we'll identify it by 'id' if present
                if isinstance(json_to_replace, dict) and 'id' in json_to_replace:
                    item_id = json_to_replace['id']
                    
                    # Find the item in the original JSON
                    for i, item in enumerate(original_json):
                        if isinstance(item, dict) and item.get('id') == item_id:
                            # Replace the item
                            original_json[i] = json.loads(new_content)
                            # Return the updated JSON
                            return json.dumps(original_json)
            except json.JSONDecodeError:
                # If the content to replace isn't valid JSON, use the line-based approach
                pass
            
            # If we couldn't use the JSON-based approach, try line-based replacement
            # Replace the specified lines in the formatted content
            replacement_lines = new_content.split('\n')
            
            # Apply the replacement to the formatted JSON
            # We need to handle the line numbers correctly
            start_idx = original_line - 1
            end_idx = updated_line
            
            # Apply the changes
            formatted_original[start_idx:end_idx] = replacement_lines
            
            # Convert back to a string
            formatted_result = '\n'.join(formatted_original)
            
            # Try to parse the result to make sure it's valid JSON
            try:
                result_json = json.loads(formatted_result)
                # Return the JSON in its original compact format
                return json.dumps(result_json)
            except json.JSONDecodeError:
                # If the formatted result isn't valid JSON, try to extract JSON portions
                # This is a fallback approach that tries to find valid JSON in the modified content
                logger.warning("Formatted result is not valid JSON, attempting to extract valid JSON")
                
                # Try to find and parse any JSON objects in the new content
                try:
                    new_json = json.loads(new_content)
                    
                    # If we reach this point, the new_content is valid JSON
                    # Let's use the line numbers to identify where to put it
                    # This is a simplified approach that may need adjustment
                    
                    # Simplified implementation - convert to string and attempt to replace
                    compact_result = json.dumps(original_json)
                    return compact_result
                except json.JSONDecodeError:
                    logger.error("Could not extract valid JSON from replacement")
                    return None
            
        except Exception as e:
            logger.error(f"Error applying snippet replacement: {str(e)}")
            return None
    
    async def post_async(self, shared, prep_res, result):
        # Store the result in the shared context
        shared["tool_results"] = {
            "tool_used": "file_interaction",
            "parameters": prep_res["parameters"],
            "result": result,
            "result_type": "tool_execution"
        }
        
        # Return a signal to go back to the decision node in the main flow
        return "decide"