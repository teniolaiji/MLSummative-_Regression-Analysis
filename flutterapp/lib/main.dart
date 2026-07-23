import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:http/http.dart' as http;

void main() => runApp(const PovertyApp());

// ---------------------------------------------------------------------------
// Design tokens
//   brand      deep indigo, the app's chrome and primary action
//   canvas     tinted light ground so white surfaces lift off it
//   scale[]    sequential severity ramp, borrowed from thematic map legends:
//              pale amber for low poverty through deep red for severe
// ---------------------------------------------------------------------------
const kBrand = Color(0xFF2E2A4F);
const kBrandSoft = Color(0xFF474184);
const kCanvas = Color(0xFFF3F2F7);
const kSurface = Colors.white;
const kBorder = Color(0xFFE3E1EC);
const kText = Color(0xFF1E1B2E);
const kMuted = Color(0xFF767284);
const kUnfilled = Color(0xFFE6E3E0);

const List<Color> kScale = [
  Color(0xFFF6C48A),
  Color(0xFFEE9A54),
  Color(0xFFDE6A3F),
  Color(0xFFC2412F),
  Color(0xFF8E2320),
];

Color severityColor(double v) {
  if (v < 5) return kScale[0];
  if (v < 15) return kScale[1];
  if (v < 30) return kScale[2];
  if (v < 50) return kScale[3];
  return kScale[4];
}

String severityLabel(double v) {
  if (v < 5) return 'Low';
  if (v < 15) return 'Moderate';
  if (v < 30) return 'Elevated';
  if (v < 50) return 'High';
  return 'Severe';
}

const List<BoxShadow> kLift = [
  BoxShadow(color: Color(0x0F1E1B2E), blurRadius: 14, offset: Offset(0, 4)),
];

TextStyle numStyle({
  double size = 17,
  FontWeight weight = FontWeight.w600,
  Color color = kText,
}) =>
    TextStyle(
      fontFamily: 'monospace',
      fontFamilyFallback: const ['monospace', 'Courier New'],
      fontSize: size,
      fontWeight: weight,
      color: color,
    );

class PovertyApp extends StatelessWidget {
  const PovertyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Headcount',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        useMaterial3: true,
        scaffoldBackgroundColor: kCanvas,
        colorScheme: ColorScheme.fromSeed(seedColor: kBrand),
        textTheme:
            const TextTheme().apply(bodyColor: kText, displayColor: kText),
      ),
      home: const PredictionPage(),
    );
  }
}

// ---------------------------------------------------------------------------
// Indicators. Ranges mirror the API's Pydantic constraints.
// ---------------------------------------------------------------------------
class Indicator {
  final String key;
  final String label;
  final String unit;
  final double min;
  final double max;

  const Indicator(this.key, this.label, this.unit, this.min, this.max);
}

class IndicatorGroup {
  final String title;
  final IconData icon;
  final List<Indicator> items;

  const IndicatorGroup(this.title, this.icon, this.items);
}

const List<IndicatorGroup> kGroups = [
  IndicatorGroup('Access', Icons.bolt_outlined, [
    Indicator('electricity_access', 'Electricity access', '%', 0, 100),
    Indicator('internet_users', 'Internet users', '%', 0, 100),
    Indicator('urban_population', 'Urban population', '%', 0, 100),
  ]),
  IndicatorGroup('Economy', Icons.show_chart, [
    Indicator('gdp_per_capita', 'GDP per capita', 'US\$', 0, 200000),
    Indicator('gdp_growth', 'GDP growth', '%/yr', -50, 50),
    Indicator('inflation', 'Inflation', '%/yr', -10, 100),
    Indicator('agri_value_added', 'Agriculture share', '% GDP', 0, 100),
  ]),
  IndicatorGroup('Human development', Icons.favorite_outline, [
    Indicator('health_expenditure', 'Health spending', '% GDP', 0, 100),
    Indicator('education_expenditure', 'Education spending', '% GDP', 0, 100),
    Indicator('life_expectancy', 'Life expectancy', 'yrs', 20, 90),
  ]),
];

const List<String> kRegions = [
  'East Asia & Pacific',
  'Europe & Central Asia',
  'Latin America & Caribbean',
  'Middle East, North Africa, Afghanistan & Pakistan',
  'North America',
  'South Asia',
  'Sub-Saharan Africa',
];

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------
class PredictionPage extends StatefulWidget {
  const PredictionPage({super.key});

  @override
  State<PredictionPage> createState() => _PredictionPageState();
}

class _PredictionPageState extends State<PredictionPage> {
  static const String apiUrl =
      'https://mlsummative-regression-analysis.onrender.com/predict';

  final _formKey = GlobalKey<FormState>();
  final Map<String, TextEditingController> _controllers = {};
  String _region = kRegions.first;

  bool _loading = false;
  double? _result;
  String? _error;

  List<Indicator> get _all => kGroups.expand((g) => g.items).toList();

  int get _filledCount =>
      _all.where((i) => _controllers[i.key]!.text.trim().isNotEmpty).length;

  @override
  void initState() {
    super.initState();
    for (final i in _all) {
      final c = TextEditingController();
      c.addListener(() => setState(() {}));
      _controllers[i.key] = c;
    }
  }

  @override
  void dispose() {
    for (final c in _controllers.values) {
      c.dispose();
    }
    super.dispose();
  }

  Future<void> _predict() async {
    FocusScope.of(context).unfocus();
    setState(() {
      _result = null;
      _error = null;
    });

    if (!_formKey.currentState!.validate()) {
      setState(
          () => _error = 'Some values are missing or outside their range.');
      return;
    }

    setState(() => _loading = true);

    final body = <String, dynamic>{'region': _region};
    for (final i in _all) {
      body[i.key] = double.parse(_controllers[i.key]!.text.trim());
    }

    try {
      final res = await http
          .post(
            Uri.parse(apiUrl),
            headers: {'Content-Type': 'application/json'},
            body: jsonEncode(body),
          )
          .timeout(const Duration(seconds: 60));

      if (res.statusCode == 200) {
        final data = jsonDecode(res.body);
        setState(() =>
            _result = (data['predicted_poverty_ratio'] as num).toDouble());
      } else if (res.statusCode == 422) {
        setState(() => _error =
            'The server rejected these values. Check each range and try again.');
      } else {
        setState(
            () => _error = 'Request failed with status ${res.statusCode}.');
      }
    } catch (_) {
      setState(() => _error =
          'No response from the server. It may be waking from idle, wait a moment and try again.');
    } finally {
      setState(() => _loading = false);
    }
  }

  void _clear() {
    for (final c in _controllers.values) {
      c.clear();
    }
    setState(() {
      _region = kRegions.first;
      _result = null;
      _error = null;
    });
  }

  @override
  Widget build(BuildContext context) {
    final total = _all.length;
    return Scaffold(
      body: Form(
        key: _formKey,
        child: CustomScrollView(
          slivers: [
            SliverToBoxAdapter(
              child: _Header(filled: _filledCount, total: total),
            ),
            SliverPadding(
              padding: const EdgeInsets.fromLTRB(16, 20, 16, 40),
              sliver: SliverList(
                delegate: SliverChildListDelegate([
                  ...kGroups.map((g) => _GroupCard(
                        group: g,
                        controllers: _controllers,
                      )),
                  _RegionCard(
                    value: _region,
                    onChanged: (v) => setState(() => _region = v),
                  ),
                  const SizedBox(height: 8),
                  _PrimaryButton(
                    label: 'Predict poverty ratio',
                    busy: _loading,
                    onTap: _loading ? null : _predict,
                  ),
                  const SizedBox(height: 10),
                  Center(
                    child: TextButton(
                      onPressed: _loading ? null : _clear,
                      child: const Text('Reset all fields',
                          style: TextStyle(color: kMuted, fontSize: 13)),
                    ),
                  ),
                  const SizedBox(height: 14),
                  _Readout(result: _result, error: _error),
                ]),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

// ---------------------------------------------------------------------------
// Header: brand block with live completion progress
// ---------------------------------------------------------------------------
class _Header extends StatelessWidget {
  final int filled;
  final int total;

  const _Header({required this.filled, required this.total});

  @override
  Widget build(BuildContext context) {
    final progress = total == 0 ? 0.0 : filled / total;
    return Container(
      width: double.infinity,
      padding: EdgeInsets.fromLTRB(
          20, MediaQuery.of(context).padding.top + 22, 20, 22),
      decoration: const BoxDecoration(
        gradient: LinearGradient(
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
          colors: [kBrand, kBrandSoft],
        ),
        borderRadius: BorderRadius.only(
          bottomLeft: Radius.circular(24),
          bottomRight: Radius.circular(24),
        ),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Container(
                padding: const EdgeInsets.all(9),
                decoration: BoxDecoration(
                  color: Colors.white.withValues(alpha: 0.14),
                  borderRadius: BorderRadius.circular(11),
                ),
                child: const Icon(Icons.groups_outlined,
                    color: Colors.white, size: 21),
              ),
              const SizedBox(width: 12),
              const Text(
                'Headcount',
                style: TextStyle(
                  color: Colors.white,
                  fontSize: 23,
                  fontWeight: FontWeight.w700,
                  letterSpacing: -0.3,
                ),
              ),
            ],
          ),
          const SizedBox(height: 12),
          Text(
            'Estimate the share of a population living below \$3.00 a day.',
            style: TextStyle(
              color: Colors.white.withValues(alpha: 0.78),
              fontSize: 13.5,
              height: 1.45,
            ),
          ),
          const SizedBox(height: 20),
          Row(
            children: [
              Expanded(
                child: ClipRRect(
                  borderRadius: BorderRadius.circular(3),
                  child: TweenAnimationBuilder<double>(
                    tween: Tween(begin: 0, end: progress),
                    duration: const Duration(milliseconds: 300),
                    builder: (_, v, __) => LinearProgressIndicator(
                      value: v,
                      minHeight: 5,
                      backgroundColor: Colors.white.withValues(alpha: 0.18),
                      valueColor:
                          const AlwaysStoppedAnimation(Color(0xFFF6C48A)),
                    ),
                  ),
                ),
              ),
              const SizedBox(width: 12),
              Text(
                '$filled/$total',
                style: numStyle(
                    size: 12.5, color: Colors.white.withValues(alpha: 0.9)),
              ),
            ],
          ),
        ],
      ),
    );
  }
}

// ---------------------------------------------------------------------------
// A card of related indicators
// ---------------------------------------------------------------------------
class _GroupCard extends StatelessWidget {
  final IndicatorGroup group;
  final Map<String, TextEditingController> controllers;

  const _GroupCard({required this.group, required this.controllers});

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: const EdgeInsets.only(bottom: 14),
      decoration: BoxDecoration(
        color: kSurface,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: kBorder),
        boxShadow: kLift,
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Padding(
            padding: const EdgeInsets.fromLTRB(16, 15, 16, 6),
            child: Row(
              children: [
                Icon(group.icon, size: 17, color: kBrandSoft),
                const SizedBox(width: 9),
                Text(
                  group.title,
                  style: const TextStyle(
                    fontSize: 14.5,
                    fontWeight: FontWeight.w700,
                    color: kText,
                  ),
                ),
              ],
            ),
          ),
          ...group.items.map((i) => _IndicatorField(
                indicator: i,
                controller: controllers[i.key]!,
              )),
          const SizedBox(height: 6),
        ],
      ),
    );
  }
}

class _IndicatorField extends StatelessWidget {
  final Indicator indicator;
  final TextEditingController controller;

  const _IndicatorField({required this.indicator, required this.controller});

  String _fmt(double v) =>
      v == v.roundToDouble() ? v.toInt().toString() : v.toString();

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.fromLTRB(16, 8, 16, 4),
      child: TextFormField(
        controller: controller,
        style: numStyle(size: 16),
        cursorColor: kBrand,
        keyboardType:
            const TextInputType.numberWithOptions(decimal: true, signed: true),
        inputFormatters: [
          FilteringTextInputFormatter.allow(RegExp(r'^-?\d*\.?\d*')),
        ],
        decoration: InputDecoration(
          labelText: indicator.label,
          labelStyle: const TextStyle(fontSize: 13.5, color: kMuted),
          floatingLabelStyle:
              const TextStyle(fontSize: 12.5, color: kBrandSoft),
          helperText: '${_fmt(indicator.min)} to ${_fmt(indicator.max)}',
          helperStyle: const TextStyle(fontSize: 11, color: kMuted),
          suffixText: indicator.unit,
          suffixStyle: numStyle(size: 12.5, color: kMuted),
          filled: true,
          fillColor: kCanvas.withValues(alpha: 0.55),
          isDense: true,
          contentPadding:
              const EdgeInsets.symmetric(horizontal: 14, vertical: 14),
          border: OutlineInputBorder(
            borderRadius: BorderRadius.circular(11),
            borderSide: const BorderSide(color: kBorder),
          ),
          enabledBorder: OutlineInputBorder(
            borderRadius: BorderRadius.circular(11),
            borderSide: const BorderSide(color: kBorder),
          ),
          focusedBorder: OutlineInputBorder(
            borderRadius: BorderRadius.circular(11),
            borderSide: const BorderSide(color: kBrandSoft, width: 1.6),
          ),
          errorBorder: OutlineInputBorder(
            borderRadius: BorderRadius.circular(11),
            borderSide: const BorderSide(color: Color(0xFFC2412F)),
          ),
          focusedErrorBorder: OutlineInputBorder(
            borderRadius: BorderRadius.circular(11),
            borderSide: const BorderSide(color: Color(0xFFC2412F), width: 1.6),
          ),
          errorStyle: const TextStyle(fontSize: 11, color: Color(0xFFC2412F)),
        ),
        validator: (value) {
          if (value == null || value.trim().isEmpty) return 'Required';
          final n = double.tryParse(value.trim());
          if (n == null) return 'Enter a number';
          if (n < indicator.min || n > indicator.max) {
            return 'Must be ${_fmt(indicator.min)} to ${_fmt(indicator.max)}';
          }
          return null;
        },
      ),
    );
  }
}

// ---------------------------------------------------------------------------
// Region picker
// ---------------------------------------------------------------------------
class _RegionCard extends StatelessWidget {
  final String value;
  final ValueChanged<String> onChanged;

  const _RegionCard({required this.value, required this.onChanged});

  Future<void> _pick(BuildContext context) async {
    final picked = await showModalBottomSheet<String>(
      context: context,
      backgroundColor: kSurface,
      isScrollControlled: true, // lets the sheet grow past 50% height
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(22)),
      ),
      builder: (_) => SafeArea(
        child: ConstrainedBox(
          constraints: BoxConstraints(
            maxHeight: MediaQuery.of(context).size.height * 0.75,
          ),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Center(
                child: Container(
                  width: 36,
                  height: 4,
                  margin: const EdgeInsets.only(top: 10, bottom: 14),
                  decoration: BoxDecoration(
                    color: kBorder,
                    borderRadius: BorderRadius.circular(2),
                  ),
                ),
              ),
              const Padding(
                padding: EdgeInsets.fromLTRB(20, 0, 20, 8),
                child: Text('Select region',
                    style:
                        TextStyle(fontSize: 15, fontWeight: FontWeight.w700)),
              ),
              Flexible(
                // scrolls if the list is tall
                child: ListView(
                  shrinkWrap: true,
                  children: kRegions
                      .map((r) => ListTile(
                            onTap: () => Navigator.pop(context, r),
                            title:
                                Text(r, style: const TextStyle(fontSize: 14)),
                            trailing: r == value
                                ? const Icon(Icons.check_circle,
                                    color: kBrandSoft, size: 20)
                                : null,
                          ))
                      .toList(),
                ),
              ),
              const SizedBox(height: 8),
            ],
          ),
        ),
      ),
    );
    if (picked != null) onChanged(picked);
  }

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: const EdgeInsets.only(bottom: 18),
      decoration: BoxDecoration(
        color: kSurface,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: kBorder),
        boxShadow: kLift,
      ),
      child: InkWell(
        borderRadius: BorderRadius.circular(16),
        onTap: () => _pick(context),
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Row(
            children: [
              const Icon(Icons.public, size: 17, color: kBrandSoft),
              const SizedBox(width: 9),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Text('Region',
                        style: TextStyle(fontSize: 12, color: kMuted)),
                    const SizedBox(height: 3),
                    Text(value,
                        style: const TextStyle(
                            fontSize: 14, fontWeight: FontWeight.w600)),
                  ],
                ),
              ),
              const Icon(Icons.expand_more, color: kMuted),
            ],
          ),
        ),
      ),
    );
  }
}

// ---------------------------------------------------------------------------
// Primary action
// ---------------------------------------------------------------------------
class _PrimaryButton extends StatelessWidget {
  final String label;
  final bool busy;
  final VoidCallback? onTap;

  const _PrimaryButton({required this.label, this.busy = false, this.onTap});

  @override
  Widget build(BuildContext context) {
    final enabled = onTap != null;
    return SizedBox(
      width: double.infinity,
      height: 54,
      child: Material(
        color: Colors.transparent,
        child: InkWell(
          borderRadius: BorderRadius.circular(14),
          onTap: onTap,
          child: Ink(
            decoration: BoxDecoration(
              gradient: enabled
                  ? const LinearGradient(colors: [kBrand, kBrandSoft])
                  : null,
              color: enabled ? null : kMuted,
              borderRadius: BorderRadius.circular(14),
              boxShadow: enabled
                  ? const [
                      BoxShadow(
                          color: Color(0x332E2A4F),
                          blurRadius: 16,
                          offset: Offset(0, 6))
                    ]
                  : null,
            ),
            child: Center(
              child: busy
                  ? const SizedBox(
                      width: 20,
                      height: 20,
                      child: CircularProgressIndicator(
                          strokeWidth: 2.2, color: Colors.white),
                    )
                  : Text(
                      label,
                      style: const TextStyle(
                        color: Colors.white,
                        fontSize: 15.5,
                        fontWeight: FontWeight.w700,
                      ),
                    ),
            ),
          ),
        ),
      ),
    );
  }
}

// ---------------------------------------------------------------------------
// Readout: population grid coloured by severity
// ---------------------------------------------------------------------------
class _Readout extends StatelessWidget {
  final double? result;
  final String? error;

  const _Readout({this.result, this.error});

  @override
  Widget build(BuildContext context) {
    if (error != null) {
      return Container(
        width: double.infinity,
        padding: const EdgeInsets.all(16),
        decoration: BoxDecoration(
          color: const Color(0xFFFDEEEA),
          borderRadius: BorderRadius.circular(14),
          border: Border.all(color: const Color(0xFFF0C4B8)),
        ),
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Icon(Icons.error_outline, color: Color(0xFFC2412F), size: 19),
            const SizedBox(width: 11),
            Expanded(
              child: Text(error!,
                  style: const TextStyle(
                      color: Color(0xFF8E2320), fontSize: 13, height: 1.45)),
            ),
          ],
        ),
      );
    }

    if (result == null) {
      return Container(
        width: double.infinity,
        padding: const EdgeInsets.all(20),
        decoration: BoxDecoration(
          color: kSurface,
          borderRadius: BorderRadius.circular(16),
          border: Border.all(color: kBorder),
        ),
        child: const Row(
          children: [
            Icon(Icons.insights_outlined, color: kMuted, size: 19),
            SizedBox(width: 11),
            Expanded(
              child: Text(
                'Fill in every indicator and pick a region to see the estimate.',
                style: TextStyle(color: kMuted, fontSize: 13, height: 1.45),
              ),
            ),
          ],
        ),
      );
    }

    final color = severityColor(result!);
    return TweenAnimationBuilder<double>(
      tween: Tween(begin: 0, end: result!),
      duration: const Duration(milliseconds: 900),
      curve: Curves.easeOutCubic,
      builder: (context, value, _) {
        final filled = value.round().clamp(0, 100);
        return Container(
          width: double.infinity,
          padding: const EdgeInsets.all(20),
          decoration: BoxDecoration(
            color: kSurface,
            borderRadius: BorderRadius.circular(18),
            border: Border.all(color: kBorder),
            boxShadow: kLift,
          ),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        const Text('Estimated headcount ratio',
                            style: TextStyle(fontSize: 12.5, color: kMuted)),
                        const SizedBox(height: 8),
                        Row(
                          crossAxisAlignment: CrossAxisAlignment.baseline,
                          textBaseline: TextBaseline.alphabetic,
                          children: [
                            Text(value.toStringAsFixed(1),
                                style: numStyle(
                                    size: 42,
                                    weight: FontWeight.w700,
                                    color: color)),
                            const SizedBox(width: 4),
                            Text('%',
                                style: numStyle(
                                    size: 19,
                                    weight: FontWeight.w600,
                                    color: color)),
                          ],
                        ),
                      ],
                    ),
                  ),
                  Container(
                    padding:
                        const EdgeInsets.symmetric(horizontal: 11, vertical: 6),
                    decoration: BoxDecoration(
                      color: color.withValues(alpha: 0.15),
                      borderRadius: BorderRadius.circular(20),
                    ),
                    child: Text(
                      severityLabel(result!),
                      style: TextStyle(
                        color: color,
                        fontSize: 12,
                        fontWeight: FontWeight.w700,
                      ),
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 20),
              _PopulationGrid(filled: filled, color: color),
              const SizedBox(height: 14),
              Text(
                'About $filled in every 100 people live below \$3.00 a day.',
                style: const TextStyle(
                    fontSize: 12.5, color: kMuted, height: 1.45),
              ),
            ],
          ),
        );
      },
    );
  }
}

class _PopulationGrid extends StatelessWidget {
  final int filled;
  final Color color;

  const _PopulationGrid({required this.filled, required this.color});

  @override
  Widget build(BuildContext context) {
    return Column(
      children: List.generate(10, (row) {
        return Padding(
          padding: const EdgeInsets.only(bottom: 6),
          child: Row(
            children: List.generate(10, (col) {
              final index = row * 10 + col;
              final isFilled = index < filled;
              return Expanded(
                child: Padding(
                  padding: const EdgeInsets.only(right: 6),
                  child: AspectRatio(
                    aspectRatio: 1,
                    child: Container(
                      decoration: BoxDecoration(
                        color: isFilled ? color : kUnfilled,
                        borderRadius: BorderRadius.circular(3),
                      ),
                    ),
                  ),
                ),
              );
            }),
          ),
        );
      }),
    );
  }
}
