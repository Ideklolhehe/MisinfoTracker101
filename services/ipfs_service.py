"""
IPFS Service for decentralized content storage and publishing.

This module provides functionality for storing and retrieving content
from the InterPlanetary File System (IPFS) network.
"""

import os
import json
import logging
import ipfshttpclient
import requests
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

class IPFSService:
    """Service for interacting with IPFS for decentralized storage and publishing."""
    
    def __init__(self, connect_to_daemon: bool = True):
        """
        Initialize the IPFS service.
        
        Args:
            connect_to_daemon: Whether to connect to a local IPFS daemon.
                If False, will use HTTP gateways for read operations.
        """
        self.client = None
        self.connected = False
        self.public_gateways = [
            "https://ipfs.io/ipfs/",
            "https://gateway.ipfs.io/ipfs/",
            "https://cloudflare-ipfs.com/ipfs/"
        ]
        
        # Try to connect to the IPFS daemon if requested
        if connect_to_daemon:
            try:
                self.client = ipfshttpclient.connect()
                self.connected = True
                logger.info("Connected to IPFS daemon")
            except Exception as e:
                logger.warning(f"Failed to connect to IPFS daemon: {e}")
                logger.info("Using HTTP gateways for read operations")
    
    def add_content(self, content: Union[str, bytes], metadata: Optional[Dict] = None) -> Optional[str]:
        """
        Add content to IPFS network.
        
        Args:
            content: Content to add (string or bytes)
            metadata: Optional metadata to store with the content
        
        Returns:
            IPFS hash (CID) of the content or None if failed
        """
        if not self.connected or not self.client:
            logger.error("Not connected to IPFS daemon, cannot add content")
            return None
        
        try:
            # Prepare content with metadata if provided
            if metadata:
                if isinstance(content, str):
                    content_bytes = content.encode('utf-8')
                else:
                    content_bytes = content
                
                # Create a JSON structure with content and metadata
                wrapped_content = {
                    "content": content_bytes.decode('utf-8') if isinstance(content_bytes, bytes) else content_bytes,
                    "metadata": metadata,
                    "timestamp": datetime.utcnow().isoformat()
                }
                
                # Add the JSON structure to IPFS
                result = self.client.add_json(wrapped_content)
            else:
                # Add raw content to IPFS
                if isinstance(content, str):
                    result = self.client.add_str(content)
                else:
                    result = self.client.add_bytes(content)
            
            logger.info(f"Content added to IPFS with hash: {result}")
            return result
        except Exception as e:
            logger.error(f"Failed to add content to IPFS: {e}")
            return None
    
    def add_file(self, file_path: str, metadata: Optional[Dict] = None) -> Optional[str]:
        """
        Add a file to IPFS network.
        
        Args:
            file_path: Path to the file to add
            metadata: Optional metadata to store with the file
        
        Returns:
            IPFS hash (CID) of the file or None if failed
        """
        if not self.connected or not self.client:
            logger.error("Not connected to IPFS daemon, cannot add file")
            return None
        
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                logger.error(f"File not found: {file_path}")
                return None
            
            # If metadata is provided, create a wrapper
            if metadata:
                # Read file content
                with open(file_path, 'rb') as f:
                    file_content = f.read()
                
                # Determine if content is text or binary
                try:
                    file_text = file_content.decode('utf-8')
                    is_binary = False
                except UnicodeDecodeError:
                    file_text = None
                    is_binary = True
                
                # Create metadata wrapper
                if is_binary:
                    # For binary files, just add the file directly and metadata separately
                    result = self.client.add(file_path)
                    file_hash = result['Hash']
                    
                    # Add metadata with reference to file
                    metadata_wrapper = {
                        "file_hash": file_hash,
                        "metadata": metadata,
                        "filename": os.path.basename(file_path),
                        "timestamp": datetime.utcnow().isoformat(),
                        "binary": True
                    }
                    metadata_hash = self.client.add_json(metadata_wrapper)
                    
                    # Return the metadata hash as the main reference
                    return metadata_hash
                else:
                    # For text files, create a JSON structure
                    wrapped_content = {
                        "content": file_text,
                        "metadata": metadata,
                        "filename": os.path.basename(file_path),
                        "timestamp": datetime.utcnow().isoformat(),
                        "binary": False
                    }
                    return self.client.add_json(wrapped_content)
            else:
                # Just add the file directly
                result = self.client.add(file_path)
                return result['Hash']
        
        except Exception as e:
            logger.error(f"Failed to add file to IPFS: {e}")
            return None
    
    def get_content(self, ipfs_hash: str, timeout: int = 30) -> Optional[Union[str, Dict]]:
        """
        Retrieve content from IPFS network.
        
        Args:
            ipfs_hash: IPFS hash (CID) of the content
            timeout: Timeout in seconds for the request
        
        Returns:
            Content as string or parsed JSON object, or None if failed
        """
        if self.connected and self.client:
            try:
                # Try to get content using the IPFS client
                content = self.client.cat(ipfs_hash, timeout=timeout)
                
                # Try to parse as JSON
                try:
                    return json.loads(content)
                except:
                    # Return as raw content (string)
                    return content.decode('utf-8')
            except Exception as e:
                logger.warning(f"Failed to get content using IPFS client: {e}")
                # Fall back to HTTP gateways
        
        # Try HTTP gateways
        for gateway in self.public_gateways:
            try:
                url = f"{gateway}{ipfs_hash}"
                response = requests.get(url, timeout=timeout)
                if response.status_code == 200:
                    # Try to parse as JSON
                    try:
                        return response.json()
                    except:
                        # Return as raw content
                        return response.text
            except Exception as e:
                logger.warning(f"Failed to get content from gateway {gateway}: {e}")
                continue
        
        logger.error(f"Failed to get content with hash {ipfs_hash} from any source")
        return None
    
    def pin_content(self, ipfs_hash: str) -> bool:
        """
        Pin content to ensure it stays on the IPFS network.
        
        Args:
            ipfs_hash: IPFS hash (CID) of the content to pin
        
        Returns:
            True if successful, False otherwise
        """
        if not self.connected or not self.client:
            logger.error("Not connected to IPFS daemon, cannot pin content")
            return False
        
        try:
            self.client.pin.add(ipfs_hash)
            logger.info(f"Content pinned: {ipfs_hash}")
            return True
        except Exception as e:
            logger.error(f"Failed to pin content: {e}")
            return False
    
    def publish_to_ipns(self, ipfs_hash: str, key_name: str = "self") -> Optional[str]:
        """
        Publish IPFS hash to IPNS for mutable content addressing.
        
        Args:
            ipfs_hash: IPFS hash (CID) to publish
            key_name: Name of the IPNS key to use
        
        Returns:
            IPNS name if successful, None otherwise
        """
        if not self.connected or not self.client:
            logger.error("Not connected to IPFS daemon, cannot publish to IPNS")
            return None
        
        try:
            result = self.client.name.publish(ipfs_hash, key=key_name)
            ipns_name = result.get('Name')
            logger.info(f"Published {ipfs_hash} to IPNS: {ipns_name}")
            return ipns_name
        except Exception as e:
            logger.error(f"Failed to publish to IPNS: {e}")
            return None
    
    def resolve_ipns(self, ipns_name: str, timeout: int = 30) -> Optional[str]:
        """
        Resolve IPNS name to IPFS hash.
        
        Args:
            ipns_name: IPNS name to resolve
            timeout: Timeout in seconds
        
        Returns:
            IPFS hash if successful, None otherwise
        """
        if self.connected and self.client:
            try:
                start_time = time.time()
                while time.time() - start_time < timeout:
                    try:
                        result = self.client.name.resolve(ipns_name)
                        ipfs_path = result.get('Path')
                        if ipfs_path:
                            # Extract hash from path (/ipfs/HASH)
                            ipfs_hash = ipfs_path.split('/')[-1]
                            logger.info(f"Resolved IPNS {ipns_name} to {ipfs_hash}")
                            return ipfs_hash
                    except Exception as e:
                        # Wait and retry
                        time.sleep(1)
                
                logger.error(f"Timed out resolving IPNS name: {ipns_name}")
                return None
            except Exception as e:
                logger.error(f"Failed to resolve IPNS name: {e}")
                return None
        
        # Try HTTP gateways
        for gateway_base in self.public_gateways:
            gateway = gateway_base.replace('/ipfs/', '/ipns/')
            try:
                url = f"{gateway}{ipns_name}"
                response = requests.get(url, timeout=timeout, allow_redirects=False)
                if response.status_code == 302:
                    # Get the redirect location
                    location = response.headers.get('Location')
                    if location and '/ipfs/' in location:
                        ipfs_hash = location.split('/ipfs/')[-1]
                        logger.info(f"Resolved IPNS {ipns_name} to {ipfs_hash} via gateway")
                        return ipfs_hash
            except Exception as e:
                logger.warning(f"Failed to resolve IPNS via gateway {gateway}: {e}")
                continue
        
        logger.error(f"Failed to resolve IPNS name {ipns_name} from any source")
        return None
    
    def create_ipns_key(self, key_name: str) -> Optional[str]:
        """
        Create a new IPNS key.
        
        Args:
            key_name: Name for the key
        
        Returns:
            IPNS name (key ID) if successful, None otherwise
        """
        if not self.connected or not self.client:
            logger.error("Not connected to IPFS daemon, cannot create IPNS key")
            return None
        
        try:
            result = self.client.key.gen(key_name, type="rsa", size=2048)
            ipns_name = result.get('Id')
            logger.info(f"Created IPNS key {key_name} with ID {ipns_name}")
            return ipns_name
        except Exception as e:
            logger.error(f"Failed to create IPNS key: {e}")
            return None
    
    def list_ipns_keys(self) -> List[Dict]:
        """
        List all IPNS keys.
        
        Returns:
            List of key info dictionaries
        """
        if not self.connected or not self.client:
            logger.error("Not connected to IPFS daemon, cannot list IPNS keys")
            return []
        
        try:
            result = self.client.key.list()
            keys = result.get('Keys', [])
            return keys
        except Exception as e:
            logger.error(f"Failed to list IPNS keys: {e}")
            return []
    
    def get_node_info(self) -> Dict:
        """
        Get information about the connected IPFS node.
        
        Returns:
            Dictionary with node information
        """
        if not self.connected or not self.client:
            return {"status": "not_connected"}
        
        try:
            id_info = self.client.id()
            return {
                "status": "connected",
                "id": id_info.get('ID'),
                "version": id_info.get('AgentVersion'),
                "protocols": id_info.get('Protocols')
            }
        except Exception as e:
            logger.error(f"Failed to get node info: {e}")
            return {"status": "error", "message": str(e)}