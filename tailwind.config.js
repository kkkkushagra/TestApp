// tailwind.config.js
module.exports = {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
    "./public/index.html"
  ],
  theme: {
    extend: {
      colors: {
        neon: {
          DEFAULT: '#A259FF',
          50: '#F3EEFF',
          100: '#E9E0FF',
          200: '#D6C2FF',
          500: '#A259FF',
          700: '#7b2ee6'
        },
        glass: '#0f0b13'
      },
      boxShadow: {
        'neon-lg': '0 10px 40px rgba(162,89,255,0.12), 0 2px 8px rgba(0,0,0,0.6)',
      },
      borderRadius: {
        'xl-plus': '14px'
      }
    }
  },
  plugins: [],
}
