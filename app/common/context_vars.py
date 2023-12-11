import contextvars

total_stats = contextvars.ContextVar('total_stats', default={})

