/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
    "./public/index.html"
  ],
  theme: {
    extend: {
      colors: {
        legal: {
          900: '#3b2f2f',
          800: '#4e3b31',
          700: '#6a4534',
          500: '#b58b61'
        }
      },
      fontFamily: {
        'merri': ['"Merriweather"', 'Georgia', 'serif']
      }
    },
  },
  plugins: [],
}
