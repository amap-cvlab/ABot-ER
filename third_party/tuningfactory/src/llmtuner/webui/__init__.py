try:
    from .interface import create_ui, create_web_demo
except ImportError as e:
    print(f"warning: failed to import webui: {e}")


__all__ = ["create_ui", "create_web_demo"]
