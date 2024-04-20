import nltk
#nltk.download('wordnet')
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
import pandas as pd


def algoritm(input_comp):

    lemmatizer = WordNetLemmatizer()

    categories_keywords = {
    'Tile Store': ['tile', 'ceramic', 'porcelain', 'grout', 'backsplash', 'flooring', 'mosaic', 'marble', 'slate', 'bathroom', 'kitchen', 'shower', 'stoneware', 'terra cotta', 'quarry'],
    'Investment Consultants & Financial Advisors': ['investment', 'financial', 'advisor', 'consultant', 'portfolio', 'retirement', 'wealth', 'management', 'stock', 'bond', 'IRA', '401k', 'mutual fund', 'estate planning', 'tax'],
    'Damage Restoration & Mold Remediation': ['damage', 'restoration', 'mold', 'remediation', 'water', 'fire', 'smoke', 'flood', 'cleanup', 'reconstruction', 'decontamination', 'mitigation', 'insurance', 'emergency', 'residential'],
    'Mortgage Brokers': ['mortgage', 'broker', 'loan', 'refinance', 'home', 'purchase', 'lender', 'rate', 'mortgaging', 'financing', 'pre-approval', 'interest', 'down payment', 'credit', 'property'],
    'Renewable energy companies': ['renewable', 'energy', 'solar', 'wind', 'biofuel', 'geothermal', 'hydroelectric', 'power', 'sustainability', 'green', 'panel', 'inverter', 'battery', 'efficiency', 'installation'],
    'Machinery parts manufacturer': ['machinery', 'parts', 'manufacturer', 'equipment', 'component', 'assembly', 'industrial', 'production', 'machine', 'mechanical', 'fabrication', 'engineering', 'metal', 'precision', 'tooling'],
    'Car Rental': ['car', 'rental', 'vehicle', 'hire', 'automobile', 'rent', 'leasing', 'airport', 'rent-a-car', 'suv', 'van', 'luxury', 'fleet', 'reservation', 'drive'],
    'Insurance - Agents, Carriers & Brokers': ['insurance', 'agent', 'broker', 'carrier', 'policy', 'coverage', 'auto', 'home', 'life', 'health', 'business', 'liability', 'property', 'premium', 'claim'],
    'Metal Fabrication Services': ['metal', 'fabrication', 'welding', 'steel', 'aluminum', 'fabricator', 'manufacturing', 'custom', 'welder', 'machining', 'production', 'structural', 'sheet metal', 'fabricate', 'metalwork'],
    'Garden Equipment & Supplies': ['garden', 'equipment', 'supply', 'lawn', 'mower', 'landscaping', 'tool', 'fertilizer', 'plant', 'glove', 'shovel', 'weed', 'trimmer', 'flower', 'outdoor'],
    'Auto Body Shops': ['auto', 'body', 'shop', 'collision', 'repair', 'paint', 'dent', 'fender', 'frame', 'bumper', 'refinish', 'estimation', 'detailing', 'restoration', 'accident'],
    'Business Consulting': ['business', 'consulting', 'consultant', 'strategy', 'management', 'advisory', 'development', 'consultancy', 'growth', 'solution', 'analysis', 'startup', 'entrepreneur', 'planning', 'marketing'],
    'Entertainers': ['entertainer', 'entertainment', 'performer', 'artist', 'musician', 'magician', 'actor', 'comedian', 'dj', 'singer', 'band', 'party', 'event', 'talent', 'show'],
    'Packing & Crating': ['packing', 'crating', 'shipping', 'packaging', 'box', 'crate', 'freight', 'container', 'parcel', 'moving', 'logistics', 'packing material', 'cargo', 'delivery', 'warehouse'],
    'Vending Machines': ['vending', 'machine', 'snack', 'drink', 'vend', 'dispenser', 'automated', 'coin', 'cashless', 'refreshment', 'coffee', 'soda', 'healthy', 'vending service', 'vendor'],
    'Pubs & Bars': ['pub', 'bar', 'tavern', 'drink', 'beer', 'cocktail', 'wine', 'pub food', 'happy hour', 'tap', 'brewery', 'bartender', 'alcohol', 'nightlife', 'pub grub'],
    'Sports Medicine & Physical Therapy': ['sports', 'medicine', 'physical therapy', 'rehabilitation', 'injury', 'athlete', 'exercise', 'recovery', 'physiotherapy', 'sports injury', 'muscle', 'joint', 'stretching', 'performance', 'sports medicine clinic'],
    'Tax Preparation': ['tax', 'preparation', 'accounting', 'tax return', 'IRS', 'tax consultant', 'tax filing', 'tax advisor', 'tax planning', 'tax deduction', 'tax refund', 'tax accountant', 'tax service', 'tax law', 'tax consultant'],
    'Motor Homes, Trailers & RVs - Dealers & Rentals': ['motor home', 'trailer', 'RV', 'dealer', 'rental', 'camping', 'recreational vehicle', 'motorhome rental', 'RV dealer', 'trailer rental', 'camper', 'motor home dealer', 'travel trailer', 'RV rental', 'RV park'],
    'Fruit & Vegetable - Markets & Stores': ['fruit', 'vegetable', 'market', 'store', 'produce', 'fresh', 'organic', 'farmers market', 'grocery', 'fruit store', 'vegetable store', 'fruit market', 'vegetable market', 'farm fresh', 'local produce'],
    'Dentists & Dental Clinics': ['dentist', 'dental clinic', 'dental care', 'teeth', 'oral health', 'dental implant', 'tooth extraction', 'cosmetic dentistry', 'family dentist', 'dental hygiene', 'orthodontics', 'toothache', 'root canal', 'dental cleaning', 'emergency dentist'],
    'Drywall & Insulation Contractors': ['drywall', 'insulation', 'contractor', 'construction', 'drywall installation', 'insulation contractor', 'drywall repair', 'insulation installation', 'drywall contractor', 'home improvement', 'drywall finish', 'wall insulation', 'insulation repair', 'soundproofing', 'residential insulation'],
    'Travel Agencies': ['travel', 'agency', 'travel agency', 'tour', 'trip', 'vacation', 'holiday', 'tourism', 'travel agent', 'travel package', 'tour operator', 'travel booking', 'destination', 'travel planner', 'cruise'],
    'Industrial Air Solutions': ['industrial', 'air', 'solution', 'filtration', 'air quality', 'industrial air', 'ventilation', 'dust collector', 'air purification', 'HVAC', 'clean air', 'industrial fan', 'air compressor', 'exhaust system', 'industrial vacuum'],
    'Surveying Services': ['surveying', 'surveyor', 'land survey', 'surveying service', 'property survey', 'boundary survey', 'topographic survey', 'construction survey', 'land surveyor', 'land mapping', 'civil engineering', 'surveying equipment', 'GIS', 'property boundary', 'site survey'],
    'Flooring Contractors': ['flooring', 'contractor', 'flooring installation', 'hardwood', 'carpet', 'tile', 'laminate', 'flooring company', 'flooring repair', 'vinyl', 'flooring contractor', 'flooring service', 'wood flooring', 'flooring specialist', 'flooring store'],
    'Fastener Suppliers': ['fastener', 'supplier', 'screw', 'bolt', 'nut', 'hardware', 'fastening', 'screw supplier', 'bolt supplier', 'nut supplier', 'industrial fastener', 'stainless steel fastener', 'screw manufacturer', 'nut manufacturer', 'bolt manufacturer'],
    'Painting, Plastering & Wall Covering': ['painting', 'plastering', 'wall covering', 'painter', 'paint', 'wallpaper', 'plaster', 'interior painting', 'exterior painting', 'drywall', 'stucco', 'wall covering installation', 'wallpaper installation', 'commercial painting', 'residential painting'],
    'Work Clothing & Protection Equipment': ['work clothing', 'protection equipment', 'safety gear', 'workwear', 'safety clothing', 'PPE', 'safety equipment', 'safety shoes', 'protective clothing', 'work gloves', 'safety helmet', 'work boots', 'high visibility clothing', 'safety vest', 'ear protection'],
    'Medical Supply Manufacturers': ['medical supply', 'manufacturer', 'medical equipment', 'medical device', 'healthcare', 'medical instrument', 'medical product', 'medical device manufacturer', 'surgical instrument', 'medical technology', 'medical consumable', 'medical apparatus', 'healthcare supply', 'medical equipment manufacturer', 'medical device company'],
    'Learning, Tutoring & Courses': ['learning', 'tutoring', 'course', 'education', 'tutor', 'online course', 'study', 'language learning', 'academic tutoring', 'private tutor', 'test preparation', 'math tutoring', 'language course', 'training', 'educational course'],
    'Warehousing & Storage': ['warehousing', 'storage', 'warehouse', 'logistics', 'distribution', 'inventory', 'storage facility', 'warehouse management', 'fulfillment', 'storage solutions', 'warehouse space', 'inventory management', 'shipping', 'storage unit', 'warehouse storage'],
    'Trucking and Logistics': ['trucking', 'logistics', 'freight', 'transportation', 'shipping', 'truck', 'freight forwarding', 'cargo', 'transport', 'logistics company', 'truck driver', 'delivery', 'haulage', 'freight logistics', 'trucking company'],
    'Beer & Liquor Stores': ['beer', 'liquor', 'alcohol', 'wine', 'spirits', 'beer store', 'liquor store', 'wine store', 'craft beer', 'liquor shop', 'beer selection', 'wine selection', 'alcohol store', 'beer tasting', 'liquor delivery'],
    'Auto Parts Manufacturers': ['auto parts', 'manufacturer', 'automotive parts', 'car parts', 'autoparts', 'auto component', 'auto spare parts', 'car accessories', 'auto parts manufacturer', 'automotive component', 'car parts manufacturer', 'engine parts', 'aftermarket parts', 'OEM parts', 'automotive supplier'],
    'Electric Supplies & Power Generation': ['electric supplies', 'power generation', 'electricity', 'electric equipment', 'electrical', 'power supply', 'electrical supply', 'electric components', 'power plant', 'generator', 'electric products', 'electrical equipment', 'electricity generation', 'power distribution', 'electrical components'],
    'Garbage Collection & Waste Disposal': ['garbage collection', 'waste disposal', 'waste management', 'trash collection', 'recycling', 'garbage disposal', 'waste removal', 'waste service', 'trash pickup', 'rubbish removal', 'garbage bin', 'recycling service', 'waste recycling', 'waste disposal service', 'dumpster rental'],
    'Employment Agencies & HR Consulting': ['employment agency', 'HR consulting', 'recruitment', 'staffing', 'human resources', 'job placement', 'recruiting agency', 'headhunter', 'job agency', 'employment', 'staffing', 'recruitment', 'HR', 'job', 'employment consultant'],
    'Housing Programs': ['housing', 'housing program', 'affordable housing', 'housing assistance', 'housing scheme', 'housing project', 'housing development', 'housing authority', 'housing initiative', 'housing support', 'low-income housing', 'housing solution', 'public housing', 'housing grant', 'housing subsidy'],
    'Newspapers & Magazines': ['newspaper', 'magazine', 'news', 'publication', 'journalism', 'media', 'press', 'online news', 'print media', 'news article', 'news source', 'current events', 'breaking news', 'news website', 'news publication'],
    'Investment Firms & Venture Capital': ['investment firm', 'venture capital', 'investment fund', 'private equity', 'investment management', 'investment company', 'venture capitalist', 'investment strategy', 'investment portfolio', 'investment opportunity', 'investment banking', 'capital investment', 'angel investor', 'investment advisor', 'investment partnership'],
    'Sporting Goods Store': ['sporting goods', 'sports equipment', 'sporting store', 'sports gear', 'sports store', 'athletic gear', 'sporting equipment', 'sports clothing', 'fitness equipment', 'sports apparel', 'sporting goods store', 'sports accessories', 'athletic wear', 'exercise equipment', 'sportswear'],
    'Electronical Components Manufacturing': ['electronic components', 'manufacturing', 'electronics', 'components', 'electronic manufacturing', 'electronic parts', 'PCB', 'printed circuit board', 'circuit board', 'electronics manufacturer', 'electronic assembly', 'electronic component manufacturer', 'OEM electronics', 'electronic manufacturing services', 'electronic device'],
    'Breweries': ['brewery', 'beer brewery', 'craft brewery', 'microbrewery', 'brewing company', 'beer production', 'brew pub', 'craft beer', 'brewing equipment', 'brewing process', 'beer tasting', 'brewmaster', 'brewing industry', 'homebrewing', 'brewery tour'],
    'Golf Courses & Country Clubs': ['golf course', 'golf', 'golfing', 'country club', 'membership', 'vacation', 'lesson', 'tournament', 'facility', 'pro', 'outing', 'wedding', 'restaurant'],
    'Airline Companies': ['airline', 'airline company', 'flight', 'aviation', 'air travel', 'airline ticket', 'airline reservation', 'airline booking', 'flight booking', 'airline flight', 'airline schedule', 'airline service', 'airline website', 'flight status', 'airline fare'],
    'Gas Stations':['gas station', 'fuel station', 'petrol station', 'gas pump', 'pump', 'gasoline', 'petrol', 'fueling', 'oil', 'convenience', 'prices', 'services', 'amenities', 'fuel'],
    "Pediatric Dentists": ["pediatric", "dentists", "children", "teeth", "oral", "health", "kids", "dentistry", "pediatricians", "baby", "tooth", "care", "family", "smile", "preventive"],
    "Periodontists": ["periodontists", "gum", "disease", "periodontal", "gums", "oral", "health", "teeth", "gum disease", "gingivitis", "dental", "plaque", "gingiva", "treatment", "gum health"],
    "Latin American Restaurants": ["latin", "american", "restaurant", "cuisine", "food", "mexican", "spanish", "south", "american", "dishes", "tacos", "burritos", "enchiladas", "empanadas", "salsa"],
    "YMCA Camps": ["ymca", "camps", "camping", "outdoors", "recreation", "summer", "camp", "youth", "activities", "nature", "sports", "adventure", "campers", "family", "fitness"],
    "Recruitment & Job Listing Services": ["recruitment", "job", "listing", "services", "employment", "hiring", "staffing", "career", "opportunities", "job search", "recruiters", "candidates", "job board", "resume", "interview"],
    "Greek Restaurants": ["greek", "restaurant", "cuisine", "food", "mediterranean", "greek food", "dishes", "gyros", "souvlaki", "hummus", "tzatziki", "feta", "olives", "greek salad", "spanakopita"],
    "Skydiving Center": ["skydiving", "center", "skydive", "adventure", "extreme", "sports", "skydivers", "freefall", "parachute", "jump", "tandem", "sky", "thrill", "drop zone", "airplane"],
    "Media Companies": ["media", "company", "entertainment", "production", "advertising", "film", "television", "creative", "digital", "content", "marketing", "video", "media production", "social", "studio"],
    "Used Auto Parts Store": ["used", "auto", "parts", "store", "car", "automotive", "vehicles", "replacement", "spare", "car parts", "truck", "accessories", "engines", "transmissions", "autoparts"],
    "Awards & Trophies & Plaques": ["awards", "trophies", "plaques", "recognition", "achievement", "trophies", "awards ceremony", "engraving", "medals", "trophy shop", "personalized", "honor", "prizes", "custom"],
    "Veterinary Associations": ["veterinary", "associations", "veterinarians", "animals", "pets", "veterinary medicine", "animal health", "veterinary care", "vet", "pet care", "animal welfare", "veterinary clinic", "pet owners", "animal hospital", "veterinary services"],
    "IoT": ["iot", "internet", "things", "technology", "connected", "devices", "smart", "home", "sensors", "data", "network", "innovation", "automation", "internet of things", "digital"],
    "Fur Goods": ["fur", "goods", "fur products", "fashion", "accessories", "clothing", "fur coats", "luxury", "warm", "soft", "pelts", "fur hats", "mink", "fox", "rabbit"],
    "Endodontists": ["endodontists", "root", "canal", "endodontics", "dental", "root canal", "treatment", "endodontic", "tooth", "pain", "pulp", "disease", "teeth", "endodontic treatment", "endodontist"],
    "Mealkit Delivery": ["meal", "kit", "delivery", "meal prep", "food", "service", "subscription", "meals", "recipes", "fresh", "ingredients", "convenience", "cooking", "chef", "healthy"],
    "Chimney services": ["chimney", "services", "chimney sweep", "cleaning", "fireplace", "repair", "chimney inspection", "chimney repair", "chimney cleaning", "chimney cap", "chimney maintenance", "flue", "chimney liners", "chimney company", "smoke"],
    "Crossfit Centers": ["crossfit", "fitness", "gym", "workout", "exercise", "training", "strength", "conditioning", "health", "functional", "fitness", "wod", "community", "athlete", "coach"],
    "Inns": ["inn", "lodging", "accommodation", "hotel", "bed", "breakfast", "historic", "innkeeper", "guesthouse", "hospitality", "travel", "romantic", "getaway", "cozy", "charm"],
    "Professional & Management Training": ["professional", "management", "training", "leadership", "development", "skills", "career", "certification", "coaching", "workshop", "corporate", "executive", "strategy", "communication", "business"],
    "Personal Trainer": ["personal trainer", "fitness", "exercise", "workout", "training", "health", "wellness", "nutrition", "strength", "coach", "motivation", "gym", "weight", "lifting", "cardio"],
    "Charter Schools": ["charter school", "education", "school", "students", "learning", "academics", "curriculum", "teachers", "classroom", "community", "enrollment", "public", "private", "college", "high school"],
    "Middle Eastern Restaurants": ["middle eastern", "restaurant", "cuisine", "food", "dining", "arabic", "mediterranean", "falafel", "shawarma", "hummus", "kebab", "grill", "baklava", "pita", "tabbouleh"],
    "Bicycle Rental Services": ["bicycle rental", "bike", "rental", "service", "cycling", "biking", "adventure", "tour", "rent", "outdoor", "explore", "trail", "path", "exercise", "recreation"],
    "Business Coaching": ["business coaching", "coach", "entrepreneur", "startup", "small business", "strategy", "leadership", "management", "growth", "development", "mentoring", "consulting", "success", "goal", "motivation"],
    "Currency Exchange Service": ["currency exchange", "exchange service", "foreign exchange", "money exchange", "currency", "foreign currency", "exchange rate", "travel", "foreign", "currency converter", "money", "international", "travel money", "bank", "finance"],
    "Tax Attorney": ["tax attorney", "tax law", "tax lawyer", "taxation", "IRS", "tax code", "tax planning", "tax dispute", "tax audit", "tax relief", "tax consultant", "tax advice", "tax compliance", "tax deduction", "tax return"],
    "Steakhouse Restaurants": ["steakhouse", "restaurant", "steak", "beef", "grill", "meat", "dining", "steak restaurant", "ribeye", "filet mignon", "porterhouse", "prime rib", "sirloin", "t-bone", "new york strip"],
    "Underwear & Lingerie": ["underwear", "lingerie", "intimate apparel", "bra", "panties", "boxers", "briefs", "thongs", "lingerie store", "undergarments", "sexy lingerie", "sleepwear", "corset", "lace", "silk"],
    "Content Marketing": ["content marketing", "marketing", "content creation", "digital marketing", "content strategy", "social media", "blogging", "content management", "SEO", "branding", "content writer", "online marketing", "content creation", "engagement", "audience"],
    "Wedding Venues": ["wedding venue", "venue", "wedding", "event space", "banquet hall", "wedding reception", "wedding ceremony", "outdoor wedding", "indoor wedding", "destination wedding", "wedding hall", "wedding location", "wedding planner", "romantic venue", "historic venue"],
    "Business Brokers": ["business broker", "business sale", "business buying", "business selling", "business valuation", "mergers and acquisitions", "business opportunity", "business investment", "business transfer", "exit strategy", "seller", "buyer", "business appraisal", "small business broker", "franchise"],
    "Duty Free Stores": ["duty free", "duty free shop", "duty free store", "duty free shopping", "duty free items", "tax free", "airport shopping", "travel retail", "duty free alcohol", "duty free perfume", "duty free cosmetics", "duty free cigarettes", "duty free electronics", "duty free prices", "international shopping"]
}

    input_comp = input_comp.lower().split()
    input_comp_lemmatized = [lemmatizer.lemmatize(word) for word in input_comp]

    for word in input_comp_lemmatized:
        for key, values in categories_keywords.items():
            if word in values:
                return key
    

def compute_accuracy(dataframe):

    counter = 0
    
    for i in range(len(dataframe)):
        ground_truth = dataframe['main_business_category'][i]
        company_name = dataframe['commercial_name'][i]

        prediction = algoritm(company_name)
        
        if prediction == ground_truth:
            counter += 1
    
    return counter / len(dataframe)


def obtine_business(dataframe):

    counter = 0

    lemmatizer = WordNetLemmatizer()

    for i in range(len(dataframe)):

        flag_synonym = 0

        company_name = dataframe['commercial_name'][i].lower().split()

        business_name = dataframe['main_business_category'][i].lower().split()
        
        lemm_company = [lemmatizer.lemmatize(word) for word in company_name]

        lemm_business = [lemmatizer.lemmatize(word) for word in business_name]
        
        for word in lemm_company:

            word_synset = wn.synsets(word)

            lemmatized_synonyms = [lemma for synset in word_synset for lemma in synset.lemma_names()]

            intersection_synonyms = list(set(lemm_company) & set(lemmatized_synonyms))

            if len(intersection_synonyms) > 0:

                flag_synonym = 1

        intersection_exact_match = list(set(lemm_company) & set(lemm_business))

        if len(intersection_exact_match) > 0 or flag_synonym == 1:
            
            counter += 1
    
    return counter / len(dataframe)

dataframe = pd.read_csv('tournament_hints_data.csv')

print(algoritm('Hatz'))