# cache_manager.py
import threading
import time
from typing import Dict, Optional
from database import get_cache_config, check_cache_space, CacheLocation


class CacheManager:
    """
    Manages model cache locations and monitors disk space usage.
    Runs background monitoring thread to periodically check cache usage.
    """

    def __init__(self, check_interval: int = 300):  # 5 minutes default
        self.check_interval = check_interval
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.cache_stats: Dict[str, dict] = {}
        self._lock = threading.Lock()

        # Initialize cache stats
        self.update_cache_stats()

    def start_monitoring(self):
        """Start the background monitoring thread."""
        if self.monitoring:
            return

        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        print(f"🔍 Cache monitoring started (check interval: {self.check_interval}s)")

    def stop_monitoring(self):
        """Stop the background monitoring thread."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        print("Cache monitoring stopped")

    def _monitor_loop(self):
        """Background monitoring loop."""
        while self.monitoring:
            try:
                self.update_cache_stats()
            except Exception as e:
                print(f"Error in cache monitoring: {e}")

            # Sleep in small chunks to allow quick shutdown
            for _ in range(self.check_interval):
                if not self.monitoring:
                    break
                time.sleep(1)

    def update_cache_stats(self):
        """Update cache statistics for all cache locations."""
        with self._lock:
            config = get_cache_config()

            # Check primary cache
            primary_stats = check_cache_space("primary")
            self.cache_stats["primary"] = primary_stats

            # Check secondary cache
            secondary_stats = check_cache_space("secondary")
            self.cache_stats["secondary"] = secondary_stats

            # Store config
            self.cache_stats["config"] = config

    def get_cache_stats(self) -> dict:
        """Get current cache statistics."""
        with self._lock:
            return self.cache_stats.copy()

    def check_space_for_model(self, cache_location: str, estimated_size_mb: float = 5000) -> dict:
        """
        Check if specified cache location has sufficient space for a model.

        Args:
            cache_location: "primary", "secondary", or "custom"
            estimated_size_mb: Estimated model size in MB (default: 5GB)

        Returns:
            dict with sufficient, warning, and space info
        """
        return check_cache_space(cache_location, estimated_size_mb)

    def get_recommended_cache(self, estimated_size_mb: float = 5000) -> str:
        """
        Get recommended cache location based on available space.

        Args:
            estimated_size_mb: Estimated model size in MB

        Returns:
            "primary" or "secondary" depending on available space
        """
        primary_check = self.check_space_for_model("primary", estimated_size_mb)

        if primary_check["sufficient"] and not primary_check["warning"]:
            return "primary"
        else:
            return "secondary"

    def get_cache_path(self, cache_location: str, custom_path: Optional[str] = None) -> str:
        """
        Get the actual filesystem path for a cache location.

        Args:
            cache_location: "primary", "secondary", or "custom"
            custom_path: Custom path if cache_location is "custom"

        Returns:
            Full filesystem path to cache directory
        """
        config = get_cache_config()

        if cache_location == "primary":
            return config["primary_path"]
        elif cache_location == "secondary":
            return config["secondary_path"]
        elif cache_location == "custom" and custom_path:
            return custom_path
        else:
            # Default to primary
            return config["primary_path"]

    def format_size_gb(self, size_gb: float) -> str:
        """Format size in GB with appropriate units."""
        if size_gb < 1:
            return f"{size_gb * 1024:.1f} MB"
        elif size_gb < 1000:
            return f"{size_gb:.1f} GB"
        else:
            return f"{size_gb / 1024:.2f} TB"

    def get_cache_summary(self) -> dict:
        """
        Get a formatted summary of cache status.

        Returns:
            dict with formatted cache information for display
        """
        stats = self.get_cache_stats()

        summary = {
            "primary": self._format_cache_info(stats.get("primary", {})),
            "secondary": self._format_cache_info(stats.get("secondary", {})),
            "config": stats.get("config", {}),
        }

        return summary

    def _format_cache_info(self, cache_info: dict) -> dict:
        """Format cache info for display."""
        if not cache_info:
            return {}

        return {
            "path": cache_info.get("cache_path", ""),
            "used": self.format_size_gb(cache_info.get("cache_size_gb", 0)),
            "limit": self.format_size_gb(cache_info.get("cache_limit_gb", 0)),
            "usage_percent": cache_info.get("usage_percent", 0),
            "disk_free": self.format_size_gb(cache_info.get("disk_free_gb", 0)),
            "disk_total": self.format_size_gb(cache_info.get("disk_total_gb", 0)),
            "sufficient": cache_info.get("sufficient", True),
            "warning": cache_info.get("warning", False),
        }
