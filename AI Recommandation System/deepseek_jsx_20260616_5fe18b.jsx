const AboutPage = () => (
  <div className="max-w-4xl mx-auto py-10">
    <h2 className="text-4xl font-bold mb-8">About AI Recommendation Systems</h2>
    <div className="space-y-6 text-gray-700 dark:text-gray-300">
      <p>
        Recommendation engines are at the heart of modern platforms like Netflix, Amazon, and Spotify.
        They analyze user preferences and item characteristics to suggest relevant content.
      </p>
      <div className="glass rounded-xl p-6">
        <h3 className="text-2xl font-semibold mb-4 text-primary">How Pattern Matching Works</h3>
        <p>
          Our system uses a <strong>weighted similarity scoring</strong> model. It compares selected user interests 
          with item tags and categories. Each match contributes to a final score, which determines ranking.
        </p>
      </div>
      <div className="glass rounded-xl p-6">
        <h3 className="text-2xl font-semibold mb-4 text-secondary">Similarity Logic</h3>
        <p><code>score = (interest_match × 0.6) + (category_match × 0.3) + (rating_factor × 0.1)</code></p>
        <p>This ensures items that closely match user interests are prioritized, while still considering overall quality.</p>
      </div>
      <div className="glass rounded-xl p-6">
        <h3 className="text-2xl font-semibold mb-4 text-accent">Explainable AI</h3>
        <p>
          Every recommendation includes a reason and confidence level (High/Medium/Low), making the AI 
          transparent and trustworthy.
        </p>
      </div>
    </div>
  </div>
);

export default AboutPage;