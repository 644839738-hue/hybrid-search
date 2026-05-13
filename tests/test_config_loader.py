"""
Unit tests for ConfigLoader - Configuration management.
"""

import pytest
import yaml
from pathlib import Path
from config_loader import ConfigLoader, load_config


@pytest.mark.unit
class TestConfigLoader:
    """Test suite for ConfigLoader class."""

    def test_load_valid_config(self, temp_config_file):
        """Test loading a valid configuration file."""
        config = ConfigLoader(temp_config_file)
        assert config.config is not None
        assert isinstance(config.config, dict)

    def test_missing_config_file_raises_error(self):
        """Test that missing config file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Configuration file not found"):
            ConfigLoader("nonexistent_config.yaml")

    def test_get_embedding_model(self, temp_config_file):
        """Test retrieving embedding model configuration."""
        config = ConfigLoader(temp_config_file)
        assert config.get_embedding_model() == "text-embedding-3-small"

    def test_get_chat_model(self, temp_config_file):
        """Test retrieving chat model configuration."""
        config = ConfigLoader(temp_config_file)
        assert config.get_chat_model() == "gpt-3.5-turbo"

    def test_get_chat_temperature(self, temp_config_file):
        """Test retrieving chat temperature configuration."""
        config = ConfigLoader(temp_config_file)
        assert config.get_chat_temperature() == 0.0

    def test_get_chunk_size(self, temp_config_file):
        """Test retrieving chunk size configuration."""
        config = ConfigLoader(temp_config_file)
        assert config.get_chunk_size() == 500

    def test_get_chunk_overlap(self, temp_config_file):
        """Test retrieving chunk overlap configuration."""
        config = ConfigLoader(temp_config_file)
        assert config.get_chunk_overlap() == 100

    def test_get_search_k(self, temp_config_file):
        """Test retrieving search k configuration."""
        config = ConfigLoader(temp_config_file)
        assert config.get_search_k() == 2

    def test_get_retry_config(self, temp_config_file):
        """Test retrieving retry configuration."""
        config = ConfigLoader(temp_config_file)
        retry_config = config.get_retry_config()
        assert retry_config["max_attempts"] == 2
        assert retry_config["min_wait"] == 1

    def test_load_config_convenience_function(self, temp_config_file):
        """Test convenience function for loading config."""
        config = load_config(temp_config_file)
        assert isinstance(config, ConfigLoader)
        assert config.get_chunk_size() == 500
