const STORAGE_KEY = 'mc-retirement-theme'

export function getInitialTheme() {
  try {
    const stored = localStorage.getItem(STORAGE_KEY)
    if (stored === 'light' || stored === 'dark') return stored
  } catch {
    /* ignore */
  }
  if (typeof window !== 'undefined' && window.matchMedia('(prefers-color-scheme: dark)').matches) {
    return 'dark'
  }
  return 'light'
}

export function applyTheme(theme) {
  document.documentElement.dataset.theme = theme
  document.documentElement.style.colorScheme = theme
  try {
    localStorage.setItem(STORAGE_KEY, theme)
  } catch {
    /* ignore */
  }
}

/** Read a CSS custom property from :root (resolved for current theme). */
export function cssVar(name, fallback = '') {
  if (typeof window === 'undefined') return fallback
  const value = getComputedStyle(document.documentElement).getPropertyValue(name).trim()
  return value || fallback
}
